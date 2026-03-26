# PIC2 – CMS Jet Substructure

A jet-substructure analysis workflow for CMS data based on **ROOT**, **FastJet**, and **MPI**.

This repository contains scripts to:

- build ROOT `jetTree` ntuples from CMS event files,
- compute Lund-plane observables and formation-time quantities,
- evaluate angular observables such as `\Delta\psi_{12}`,
- generate per-sample plots, frame sequences, and videos for visualization.

The codebase mixes **Python** (data preparation, orchestration, plotting, video generation) with **C++** (FastJet-based declustering and observable extraction).

---

## Repository layout

```text
PIC2-CMS-Jet-Substructure/
├── main/
│   ├── lund_plane3.cpp      # primary + secondary Lund plane observables
│   ├── tau.cpp              # formation-time and ΔR-related quantities
│   └── lund_softdrop.cpp    # SoftDrop/mMDT Lund observables
├── python/
│   ├── jets_const.py        # build jetTree from reconstructed jets
│   ├── jets_gen.py          # build jetTree from generator-level jets
│   ├── reco.py              # older RECO/AOD jetTree builder
│   ├── clear_root.py        # remove previously-added lund_/tau_ branches
│   ├── run.py               # compile and run the full C++ pipeline
│   ├── perm.py              # MPI scan of 3-constituent angular combinations
│   ├── iterative_lund.py    # tau-binned Lund-plane frame production
│   ├── iterative_psi12.py   # tau-binned Δψ12 frame production
│   ├── lund_video.py        # turn Lund frames into mp4 files
│   ├── dpsi12_video.py      # turn Δψ12 frames into mp4 files
│   ├── lund_analysis.ipynb  # notebook analysis
│   ├── tau_analysis.ipynb   # notebook analysis
│   └── fits.ipynb           # fitting/inspection notebook
├── root/                    # processed ROOT files
├── imgs/                    # static output plots
├── iterative_lund/          # generated Lund-frame sequences
├── iterative_psi12/         # generated Δψ12 frame sequences
├── slides/                  # presentation material
└── README.md
```

---

## What the pipeline does

### 1. Build a `jetTree`
The input step creates a ROOT `TTree` called `jetTree` containing jet-level and constituent-level information such as:

- `jet_pt`, `jet_eta`, `jet_phi`, `jet_mass`
- `const_pt`, `const_eta`, `const_phi`, `const_mass`

There are multiple entry points depending on the type of CMS input:

- `python/jets_const.py` for reconstructed jets (`slimmedJets`)
- `python/jets_gen.py` for generator-level jets (`slimmedGenJets`)
- `python/reco.py` for an older RECO/AOD workflow

### 2. Run the C++ observable chain
`python/run.py` compiles and runs three C++ programs over the ROOT files in `root/`:

1. `main/lund_plane3.cpp`
2. `main/tau.cpp`
3. `main/lund_softdrop.cpp`

Before that, it calls `python/clear_root.py` to remove stale `lund_*` and `tau_*` branches so files can be regenerated cleanly.

### 3. Produce visualization frames
After the ROOT files are enriched with Lund and tau observables, the iterative plotting scripts generate frame sequences:

- `python/iterative_lund.py` creates Lund-plane snapshots ordered by formation time
- `python/iterative_psi12.py` creates `\Delta\psi_{12}` distributions ordered by formation time

By default, these scripts apply a jet selection of roughly:

- `pT > 200 GeV`
- `|eta| < 1`

### 4. Turn frame sequences into videos
The scripts:

- `python/lund_video.py`
- `python/dpsi12_video.py`

use `ffmpeg` to convert PNG sequences into `.mp4` animations.

---

## Main outputs written to `jetTree`

### From `lund_plane3.cpp`
This step adds the **ungroomed** Lund-plane information, including branches such as:

- `lund_coords_x`, `lund_coords_y`
- `lund_kt`, `lund_z`, `lund_psi`, `lund_mass`
- `lund_coords_secondary_x`, `lund_coords_secondary_y`
- `lund_kt_secondary`, `lund_z_secondary`, `lund_psi_secondary`, `lund_mass_secondary`
- `lund_psi12`

### From `tau.cpp`
This step adds formation-time information:

- `tau_time`
- `tau_deltaR`

### From `lund_softdrop.cpp`
This step adds the **SoftDrop / mMDT** Lund-plane observables:

- `lund_coords_x_sd`, `lund_coords_y_sd`
- `lund_kt_sd`, `lund_z_sd`, `lund_psi_sd`, `lund_mass_sd`
- `lund_coords_secondary_x_sd`, `lund_coords_secondary_y_sd`
- `lund_kt_secondary_sd`, `lund_z_secondary_sd`, `lund_psi_secondary_sd`, `lund_mass_secondary_sd`
- `lund_psi12_sd`

---

## Requirements

### Core tools
You will need:

- Python 3
- CERN ROOT
- FastJet
- FastJet contrib
- MPI / OpenMPI
- `mpi4py`
- `ffmpeg` (for video generation)

### CMS-specific Python environment
The ntuple-building scripts (`jets_const.py`, `jets_gen.py`, `reco.py`) rely on CMS FWLite-style imports such as:

- `DataFormats.FWLite`
- `FWCore.ParameterSet.Config`
- `FWCore.PythonUtilities.LumiList`

That means they are intended to run inside a **CMSSW-compatible environment**.

---

## Build and run

### A. Build `jetTree` files
For reconstructed jets:

```bash
mpiexec -n 4 python python/jets_const.py
```

For generator-level jets:

```bash
mpiexec -n 4 python python/jets_gen.py
```

### B. Run the C++ analysis chain
This compiles the C++ programs into `build/` and processes all ROOT files found in `root/`:

```bash
mpiexec -n 6 python3 python/run.py
```

### C. Generate tau-ordered frame sequences

```bash
mpiexec -n 6 python3 python/iterative_lund.py
mpiexec -n 6 python3 python/iterative_psi12.py
```

### D. Convert frame sequences into videos

```bash
python3 python/lund_video.py
python3 python/dpsi12_video.py
```

---

## Additional analysis scripts

### `python/perm.py`
This script performs an MPI-parallel scan over 3-constituent combinations in each jet and fills angular histograms such as:

- `hist_psi`
- `hist_thetaS`
- `hist_thetaL`
- `hist_thetaL12`

It is useful for dedicated angular studies outside the main Lund pipeline.

---

## Notes and conventions

- The central ROOT object used throughout the repository is a `TTree` named `jetTree`.
- `python/run.py` assumes the input ROOT files to process are located in `root/`.
- The C++ programs open files in `UPDATE` mode and append branches to the existing tree.
- `python/clear_root.py` is meant to reset previously-generated Lund/tau branches before recomputing them.
- The iterative plotting scripts expect the tau and SoftDrop branches to already exist.
- The video scripts expect numbered PNG sequences in `iterative_lund/` or `iterative_psi12/`.

---

## Typical workflow

```bash
# 1) Build jetTree ROOT files
mpiexec -n 4 python python/jets_const.py

# 2) Move or copy produced ROOT files into root/

# 3) Run declustering + tau + SoftDrop augmentation
mpiexec -n 6 python3 python/run.py

# 4) Make ordered frame sequences
mpiexec -n 6 python3 python/iterative_lund.py
mpiexec -n 6 python3 python/iterative_psi12.py

# 5) Export videos
python3 python/lund_video.py
python3 python/dpsi12_video.py
```

---

## Current status

The repository already includes:

- C++ sources for the observable extraction chain,
- Python utilities for ntuple building, orchestration, plotting, and video generation,
- example plots in `imgs/`,
- generated frame folders such as `iterative_lund/` and `iterative_psi12/`,
- at least one processed ROOT file in `root/`.

---

## Suggested future improvements

A few things that would make the project easier to reuse:

- add a `requirements.txt` or environment setup guide,
- document the expected ROOT input format more explicitly,
- add command-line arguments for jet selections instead of hard-coded values,
- add a small test file and a minimal end-to-end example,
- describe the physics meaning of `psi`, `psi12`, and `tau_time` in more detail.
