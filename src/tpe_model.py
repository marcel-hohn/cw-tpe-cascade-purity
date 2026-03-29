from dataclasses import dataclass, field
import numpy as np
from qutip import basis, mesolve, steadystate, expect

# ------------------------------------------------------------
# physical parameters of the XX–X–G cascade system
# ------------------------------------------------------------

@dataclass
class TPEParameters:
    
    # global scaling factor used to keep frequencies in a numerically
    # convenient range for the solver (does not change physics)
    S: float = 1e-9
    
    # reduced Planck constant in eV*s
    # used to convert transition energies into angular frequencies
    hbar: float = 6.582119569e-16

    # radiative lifetimes (ns) of the quantum dot states
    # XX → X decay
    T1_B: float = 0.44
    
    # X → G decay for horizontally polarized exciton
    T1_H: float = 0.711
    
    # X → G decay for vertically polarized exciton
    T1_V: float = 0.711

    # transition energies in eV
    # fine structure splitting between H and V exciton
    omega_V_eV: float = 1.336128
    omega_H_eV: float = 1.336100
    Delta_Eb_eV: float = -0.0028
    
    # biexciton energy (includes binding energy shift)
    omega_B_eV: float = field(init=False)

    def __post_init__(self):
        self.omega_B_eV = self.omega_H_eV + self.omega_V_eV + self.Delta_Eb_eV  # Energy of |B> -> |G>

# default parameter set used in both notebooks
DEFAULT_PARAMS = TPEParameters()

# ------------------------------------------------------------
# basis states of the XX–X–G cascade system
# Hilbert space dimension = 4
#
# ordering convention:
# |0> = |XX>  biexciton state
# |1> = |X_V> vertically polarized exciton
# |2> = |X_H> horizontally polarized exciton
# |3> = |G>   ground state
#
# the chosen ordering is used consistently throughout the
# Hamiltonian and collapse operators
# ------------------------------------------------------------

B = basis(4, 0)   # |XX>
V = basis(4, 1)   # |X_V>
H = basis(4, 2)   # |X_H>
G = basis(4, 3)   # |G>

# ------------------------------------------------------------
# polarization states in the exciton subspace
#
# the exciton eigenstates |X_V> and |X_H> form a linear polarization basis.
# arbitrary linear polarization states are represented as superpositions
#
# |X(θ)> = cos(θ)|X_V> + sin(θ)|X_H>
#
# angle convention:
# V : vertical polarization
# H : horizontal polarization
# D : diagonal  (45°)
# A : anti-diagonal (-45°)
#
# these angles are used for both excitation and detection polarization
# in the simulation of polarization-dependent emission rates.
# ------------------------------------------------------------

ANGLES = {
    "V": 0.0,
    "D": np.pi / 4,
    "H": np.pi / 2,
    "A": 3* np.pi / 4,
}

def polarization_state(theta):
    # returns a normalized exciton polarization state in the
    # {|X_V>, |X_H>} subspace as a QuTiP ket vector
    return np.cos(theta) * V + np.sin(theta) * H

# ------------------------------------------------------------
# transition frequencies and laser reference frequency
#
# transition energies are specified in eV and converted to
# angular frequencies for use in the QuTiP Hamiltonian.
#
# a global scaling factor S keeps frequencies in a numerically
# convenient range without changing the physics.
#
# omega_L is chosen as half the biexciton energy, corresponding
# to resonant two-photon excitation of the biexciton state.
# ------------------------------------------------------------
def system_frequencies(params):

    # convert transition energies from eV to angular frequencies
    omega_V = params.omega_V_eV / params.hbar * params.S
    omega_H = params.omega_H_eV / params.hbar * params.S
    omega_B = params.omega_B_eV / params.hbar * params.S

    # laser frequency for resonant two-photon excitation
    omega_L = omega_B / 2

    return omega_H, omega_V, omega_B, omega_L


# ------------------------------------------------------------
# radiative decay rates
#
# T1 lifetimes are converted into decay rates gamma = 1 / T1.
# These rates enter the Lindblad collapse operators and describe
# spontaneous emission in the XX -> X and X -> G cascade.
# ------------------------------------------------------------
def decay_rates(params):
    gamma_B_total = 1 / params.T1_B
    gamma_H = 1 / params.T1_H
    gamma_V = 1 / params.T1_V

    # equal branching of the biexciton decay
    gamma_B_H = 0.5 * gamma_B_total
    gamma_B_V = 0.5 * gamma_B_total


    return gamma_B_H, gamma_B_V, gamma_H, gamma_V


# ------------------------------------------------------------
# Hamiltonian of the driven XX–X–G cascade system
#
# H0   : bare system Hamiltonian in the laboratory frame
# H_L  : coherent coupling to the laser field
#
# the excitation polarization is encoded in laser_state, i.e. as a
# superposition in the {|X_V>, |X_H>} exciton subspace.
#
# the final Hamiltonian is written in a rotating frame at the laser
# frequency omega_L. For the biexciton state, the rotating-frame shift
# is 2 * omega_L because the biexciton is driven via two-photon excitation.
# ------------------------------------------------------------
def hamiltonian(params, omega_drive, theta_exc):

    omega_H, omega_V, omega_B, omega_L = system_frequencies(params)

    # polarization state selected by the excitation angle
    laser_state = polarization_state(theta_exc)

    # bare energies of the exciton and biexciton states
    H0 = (
        omega_H * H * H.dag()
        + omega_V * V * V.dag()
        + omega_B * B * B.dag()
    )

    # coherent laser coupling between G <-> X and X <-> XX
    # projected onto the selected excitation polarization
    H_L = (
        omega_drive * (
            G * laser_state.dag()
            + laser_state * G.dag()
            + laser_state * B.dag()
            + B * laser_state.dag()
        )
    )

    H_sys = H0 + H_L

    # rotating-frame transformation:
    # exciton states shift by omega_L, biexciton by 2 * omega_L
    return H_sys - omega_L * (
        H * H.dag()
        + V * V.dag()
        + 2 * B * B.dag()
    )

# ------------------------------------------------------------
# radiative decay channels (Lindblad collapse operators)
#
# the open quantum system dynamics are modeled using the Lindblad
# master equation formalism implemented in QuTiP.
#
# included decay processes of the XX–X–G cascade:
#
# XX -> X_V
# XX -> X_H
# X_H -> G
# X_V -> G
#
# each collapse operator has the form:
#
# sqrt(gamma) * |final><initial|
#
# where gamma = 1/T1 is the radiative decay rate.
# ------------------------------------------------------------
def collapse_operators(params):

    gamma_B_H, gamma_B_V, gamma_H, gamma_V = decay_rates(params)

    c_ops = [

        # biexciton -> exciton decay channels
        np.sqrt(gamma_B_V) * V * B.dag(),
        np.sqrt(gamma_B_H) * H * B.dag(),

        # exciton -> ground state decay channels
        np.sqrt(gamma_H) * G * H.dag(),
        np.sqrt(gamma_V) * G * V.dag(),
    ]

    return c_ops

# ------------------------------------------------------------
# emission operators (measurement observables)
#
# these operators describe polarization-selective radiative
# transitions detected in the experiment.
#
# exciton emission:
# projection of the X -> G transition onto a selected detection
# polarization θ_det in the {|X_V>, |X_H>} basis.
#
# biexciton emission:
# projection of the XX -> X transition onto a selected detection
# polarization θ_det in the exciton subspace, corresponding to
# a polarization analyzer in the XX detection arm.
#
# expectation values of the form
#
# <c† c>
#
# correspond to emission intensities (photon count rates).
# ------------------------------------------------------------

def exciton_operator(theta_det, params=DEFAULT_PARAMS):

    # transition operator for polarization-selective detection
    # of the exciton photon: |G><X(θ)|
    _, _, gamma_H, gamma_V = decay_rates(params)

    return (
        np.sqrt(gamma_V) * np.cos(theta_det) * G * V.dag()
        +
        np.sqrt(gamma_H) * np.sin(theta_det) * G * H.dag()
    )


def biexciton_operator(theta_det, params=DEFAULT_PARAMS):

    # transition operator for polarization-selective detection
    # of the biexciton photon: |X(θ)><XX|
    gamma_B_H, gamma_B_V, _, _ = decay_rates(params)

    return (
        np.sqrt(gamma_B_V) * np.cos(theta_det) * V * B.dag()
        +
        np.sqrt(gamma_B_H) * np.sin(theta_det) * H * B.dag()
    )

# ------------------------------------------------------------
# numerical solvers
#
# these wrapper functions provide convenient access to the QuTiP
# master-equation solvers used throughout the notebooks.
#
# steady_state_density_matrix:
# computes the stationary density matrix ρ_ss of the driven open
# quantum system for fixed excitation strength and polarization.
#
# time_evolution:
# returns the QuTiP solver result for the time-dependent Lindblad
# evolution of the density matrix for a given initial state and
# time grid.
# ------------------------------------------------------------

def steady_state_density_matrix(
    omega_drive,
    theta_exc,
    params=DEFAULT_PARAMS
):

    # construct Hamiltonian and radiative decay channels
    H_sys = hamiltonian(params, omega_drive, theta_exc)
    c_ops = collapse_operators(params)

    # stationary solution of the Lindblad master equation
    return steadystate(H_sys, c_ops)


def time_evolution(
    omega_drive,
    theta_exc,
    rho0,
    tlist,
    params=DEFAULT_PARAMS
):

    # construct Hamiltonian and radiative decay channels
    H_sys = hamiltonian(params, omega_drive, theta_exc)
    c_ops = collapse_operators(params)

    # time evolution under the Lindblad master equation
    return mesolve(
        H_sys,
        rho0,
        tlist,
        c_ops,
    )

# ------------------------------------------------------------
# emission rates
#
# emission intensities are calculated from expectation values of
# the form
#
# <c† c>
#
# where c is the corresponding transition operator.
#
# for the exciton and biexciton channels, the detection polarization
# is set by theta_det and defines the transmission axis of the
# polarization analyzer in the respective detection arm.
# ------------------------------------------------------------

def exciton_emission_rate(rho, theta_det, params=DEFAULT_PARAMS):

    op = exciton_operator(theta_det, params)

    return expect(op.dag() * op, rho)


def biexciton_emission_rate(rho, theta_det, params=DEFAULT_PARAMS):

    op = biexciton_operator(theta_det, params)

    return expect(op.dag() * op, rho)

# ------------------------------------------------------------
# state populations
#
# these helper functions return the populations of the four
# basis states of the XX–X–G cascade system.
#
# state_populations:
# returns the populations of the biexciton, both excitons,
# and the ground state for a single density matrix rho.
#
# populations_vs_time:
# evaluates the same quantities for each density matrix in a
# time evolution result returned by mesolve.
# ------------------------------------------------------------

def state_populations(rho):

    return {
        "B": expect(B * B.dag(), rho),
        "V": expect(V * V.dag(), rho),
        "H": expect(H * H.dag(), rho),
        "G": expect(G * G.dag(), rho),
    }


def populations_vs_time(result):

    return {
        "B": np.array([expect(B * B.dag(), rho) for rho in result.states]),
        "V": np.array([expect(V * V.dag(), rho) for rho in result.states]),
        "H": np.array([expect(H * H.dag(), rho) for rho in result.states]),
        "G": np.array([expect(G * G.dag(), rho) for rho in result.states]),
    }

# ------------------------------------------------------------
# metrics
#
# power_to_rabi:
#
# converts optical excitation power into an effective Rabi
# frequency assuming
#
# Ω ∝ √P
#
# which is valid for dipole coupling to a classical field.
# ------------------------------------------------------------

def power_to_rabi(power):

    return np.sqrt(power)