This repository contains code to read ROOT trees, process jet constituents, compute angular observables, and generate Lund plane coordinates and plots. It includes Python scripts for data preparation and analysis, and a C++ program for detailed Lund-plane declustering using FastJet.

## Python scripts

- [python/perm.py](/Users/alexmendonca/Desktop/pic2/python/perm.py)
  - MPI-parallel ROOT analysis over jets and their constituents.
  - Reads `jetTree` from a ROOT file, iterates entries distributed by MPI rank, filters jets (e.g., `jet_pt < 200`), and computes:
    - 3D momentum vectors from pt/eta/phi via [`vec_from_pt_eta_phi`](/Users/alexmendonca/Desktop/pic2/python/perm.py)
    - Pairwise angles via [`angle_between`](/Users/alexmendonca/Desktop/pic2/python/perm.py)
    - Signed plane angle between three vectors via [`plane_angle_signed`](/Users/alexmendonca/Desktop/pic2/python/perm.py)
  - Fills ROOT histograms (`hist_psi`, `hist_thetaS`, `hist_thetaL`, `hist_thetaL12`) and saves plots per rank.
  - Intended for scanning 3-constituent permutations per jet and accumulating weighted distributions.

- [python/jets_cont.py](/Users/alexmendonca/Desktop/pic2/python/jets_cont.py)
  - Builds a ROOT TTree of jets and their constituents from CMS EDM events.
  - For each event:
    - Extracts PF jets and AK8 jets, saves jet-level features (pt, eta, phi, mass, b-tag).
    - Iterates jet constituents to store per-jet vectors: `const_pt`, `const_eta`, `const_phi`, `const_mass`, `const_pdgId`.
  - Provides progress prints per rank, supports limiting events with `maxEvents`.

- [python/run.py](/Users/alexmendonca/Desktop/pic2/python/run.py)
  - Likely a runner/entry-point to orchestrate analyses (e.g., invoking perm or plotting scripts). If used, check for command-line argument parsing and calls into other modules.

- [python/clear_root.py](/Users/alexmendonca/Desktop/pic2/python/clear_root.py)
  - Utility to clean or reset ROOT outputs (e.g., delete branches or clear files). Use to maintain a clean workspace between runs.

- [python/teste.py](/Users/alexmendonca/Desktop/pic2/python/teste.py)
  - Scratch/testing script. Likely used to prototype reading ROOT files, quick calculations, or verifying functions.

- [python/declustering.ipynb](/Users/alexmendonca/Desktop/pic2/python/declustering.ipynb)
  - Jupyter notebook exploring declustering, possibly mirroring the C++ Lund-plane workflow in Python for sanity checks or visualization.

- [python/plot_lund.ipynb](/Users/alexmendonca/Desktop/pic2/python/plot_lund.ipynb)
  - Jupyter notebook for plotting Lund-plane outputs saved by the C++ program. Expect histograms, scatter plots of ln(1/Δ) vs ln(k_t), z, ψ distributions.

- [python/name.ipynb](/Users/alexmendonca/Desktop/pic2/python/name.ipynb)
  - Notebook placeholder; likely experiments with naming conventions or small data checks.

## C++ program

- [main/lund_plane.cpp](/Users/alexmendonca/Desktop/pic2/main/lund_plane.cpp)
  - Reads a ROOT TTree of jets and their constituents.
  - Uses FastJet (Cambridge/Aachen) and contrib tools (SoftDrop, Lund declustering) to compute:
    - Primary Lund coordinates ln(1/Δ) and ln(k_t) for each splitting.
    - Secondary Lund plane quantities via `SecondaryLund_mMDT`.
    - SoftDrop primary and secondary planes (β, zcut, R0 parameters configurable in code).
  - Computes and stores per-splitting variables (mass, z, κ, ψ) via helper function `dic_var`.
  - Fills multiple branches:
    - Primary: `lund_coords_x`, `lund_coords_y`, `lund_delta`, `lund_kt`, `lund_z`, `lund_psi`, `lund_kappa`, `lund_mass`
    - Secondary: `lund_coords_secondary_x`, `lund_coords_secondary_y`, `lund_delta_secondary`, `lund_kt_secondary`, `lund_z_secondary`, `lund_psi_secondary`, `lund_kappa_secondary`, `lund_mass_secondary`
    - SoftDrop primary: `lund_coords_x_sd`, `lund_coords_y_sd`, `lund_delta_sd`, `lund_kt_sd`, `lund_z_sd`, `lund_psi_sd`, `lund_kappa_sd`, `lund_mass_sd`
    - SoftDrop secondary: `lund_coords_x_sd_secondary`, `lund_coords_y_sd_secondary`, `lund_delta_sd_secondary`, `lund_kt_sd_secondary`, `lund_z_sd_secondary`, `lund_psi_sd_secondary`, `lund_kappa_sd_secondary`, `lund_mass_sd_secondary`
    - Additional grouped outputs: hard/soft/preferential split tracks (`lund_hard_*`, `lund_soft_*`, `lund_pref_*`)
  - Prints periodic progress (every 1000 events).
  - Outputs vectors per event and fills branches so they can be saved to a ROOT file for downstream analysis.

## Notes

- Python scripts assume a ROOT file with a `jetTree` containing jet-level vectors and nested constituent vectors (as created by `jets_cont.py`).
- MPI support:
  - [`perm.py`](/Users/alexmendonca/Desktop/pic2/python/perm.py) distributes work across ranks using `range(rank, n_entries, size)`.
  - When running under MPI (Open MPI), prefer the VS Code Terminal and unbuffered Python (`python -u`) for live progress output.
- Angles:
  - `np.arccos` returns radians in [0, π].
  - Signed plane angle in [`plane_angle_signed`](/Users/alexmendonca/Desktop/pic2/python/perm.py) is computed by the angle between planes formed by (p1, p2) and (p1+p2, p3), with sign from the triple product.
