# Direct exciton fraction in the cw driven XX–X cascade

Numerical study of cascade purity in semiconductor quantum dots under continuous-wave two-photon excitation (TPE).

This project models the biexciton–exciton cascade in a semiconductor quantum dot under continuous-wave two-photon excitation (TPE). The goal is to quantify unwanted direct exciton population, which reduces the purity of the cascade emission and degrades two-photon interference in energy–time entanglement experiments.

The model is implemented as a four-level open quantum system and solved using a Lindblad master equation with QuTiP.

All physical parameters are explicitly defined and can be adapted to represent a specific quantum dot by inserting experimentally measured values such as transition energies, fine structure splitting, and radiative lifetimes. The framework can therefore be used to estimate cascade purity and optimize polarization configurations for realistic experimental systems.

---

## Physical motivation

Photon pairs generated via the biexciton cascade (XX → X → G) are widely used for quantum optics experiments, including energy–time entanglement measurements with a Franson interferometer.

Under continuous-wave resonant two-photon excitation, the biexciton state is coherently populated. However, polarization mixing can introduce additional excitation pathways that directly populate the exciton state. Photons emitted through these processes do not originate from the cascade and therefore contribute an incoherent background that reduces the observable entanglement visibility.

To quantify this effect, the notebook introduces the direct exciton fraction (DEF)

```
DEF = (I_X − I_XX) / I_X
```

which estimates the fraction of detected exciton emission that does not originate from the cascade channel.

---

## Model

The quantum dot is described as a four-level system

```
|XX>, |X_H>, |X_V>, |G>
```

including:

- resonant cw two-photon excitation of the biexciton
- polarization-dependent excitation coupling
- radiative decay XX → X → G
- polarization-selective detection

The steady state density matrix is obtained by solving the Lindblad master equation using QuTiP.

---

## Results

The simulations show:

- orthogonal excitation–detection polarization suppresses direct exciton emission
- the direct exciton fraction depends primarily on the relative polarization angle
- at high excitation strengths the cascade dominates the steady state emission
- polarization filtering can be used to control cascade purity under cw TPE

The notebook visualizes the DEF as a function of:

- excitation power
- detection polarization angle
- excitation polarization basis

---

## Repository structure

```
.
├── cw_tpe_direct_exciton_fraction.ipynb
├── requirements.txt
└── src
    └── tpe_model.py
```

---

## Installation

Clone the repository and create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Open the notebook:

```
cw_tpe_direct_exciton_fraction.ipynb
```

or use the Python environment directly in VS Code.

---

## Dependencies

- numpy
- scipy
- matplotlib
- qutip
- ipykernel
- jupyter

---

## Relation to energy–time entanglement experiments

The model was developed in the context of energy–time entanglement experiments based on the XX → X cascade and was used to investigate how polarization dependent detection influences the observed two-photon interference visibility in a Franson interferometer.

In such measurements, only photon pairs emitted through the cascade contribute to the interference signal. Directly excited exciton emission introduces uncorrelated background photons that reduce the observable visibility.

The simulations show that polarization mixing between excitation and detection channels can lead to a finite direct exciton contribution under cw two-photon excitation. The model therefore provides a physically motivated explanation for a polarization dependent background contribution observed in experiment.

---

## Author

Marcel Hohn
