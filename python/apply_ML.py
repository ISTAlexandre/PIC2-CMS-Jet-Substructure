# -*- coding: utf-8 -*-
"""
Apply the stage-1 (matched-vs-unmatched) and stage-2 (qqbar binary) DNN models
to every jet in a ROOT file and write two new per-jet branches:

    lund_ML_matched  = stage-1 matched score  (softmax column 1)
    lund_ML_qq       = stage-2 qqbar score    (sigmoid)

Jets that don't have a valid primary+secondary splitting (max-kt indices < 0)
get the sentinel value -1.0 in both branches, so downstream you can select
valid jets with score >= 0.

Feature extraction mirrors make_csv_ch2.py EXACTLY so the models see the same
inputs they were trained on.
"""

import ROOT
import numpy as np
import pandas as pd
from joblib import load
import tensorflow as tf
from tensorflow import keras

# ============================== config =====================================
ROOT_FILE = "root_ML/merged_ML-pb80.root"   # <-- your file (edited in place)
TREE_NAME = "jetTree"

S1_DIR   = "python/models_stage1_unmatched_vs_matched"
S1_MODEL = f"{S1_DIR}/stage1_unmatched_vs_matched_dnn_best.keras"
S1_PREP  = f"{S1_DIR}/stage1_preprocess.joblib"          # bare sklearn Pipeline

S2_DIR   = "python/models_stage2_qqbar_binary"
S2_MODEL = f"{S2_DIR}/qqbar_binary_dnn_best.keras"
S2_PREP  = f"{S2_DIR}/qqbar_binary_preprocessor.joblib"  # dict: imputer/scaler/features

SENTINEL = -1.0
MAX_EVENTS = None    # set an int to test on a few events; None = all
BATCH = 8192

# ============================== load models ================================
print("Loading models ...")
prep1 = load(S1_PREP)
try:
    s1_features = list(prep1.named_steps["imputer"].feature_names_in_)
except Exception:
    s1_features = list(prep1.steps[0][1].feature_names_in_)
m1 = keras.models.load_model(S1_MODEL, compile=False)
print("  stage-1 features:", len(s1_features))

pp2 = load(S2_PREP)
imp2, scl2, feat2 = pp2["imputer"], pp2["scaler"], pp2["features"]
m2 = keras.models.load_model(S2_MODEL, compile=False)
print("  stage-2 features:", len(feat2))

# ============================== stage-2 engineering ========================
def engineer_stage2(df):
    """Identical to the stage-2 training notebook."""
    df = df.copy()
    for col in ["pt_dispersion3", "pt_dispersion4"]:
        if col in df.columns:
            df[col] = np.log1p(df[col].clip(lower=0))
    if "kt2" in df.columns:
        df["kt2"] = np.log1p(df["kt2"].clip(lower=0))
    df["z_sum"] = df["z1"] + df["z2"]
    df["z_absdiff"] = np.abs(df["z1"] - df["z2"])
    df["z_min"] = np.minimum(df["z1"], df["z2"])
    df["z_max"] = np.maximum(df["z1"], df["z2"])
    df["z_minmax_ratio"] = df["z_min"] / (df["z_max"] + 1e-8)
    for c3, c4, name in [("N_charged3","N_charged4","N_charged"),
                         ("N_all3","N_all4","N_all"),
                         ("pt_weight3","pt_weight4","pt_weight"),
                         ("pt_dispersion3","pt_dispersion4","pt_dispersion")]:
        if c3 in df.columns and c4 in df.columns:
            v3, v4 = df[c3], df[c4]
            vmin, vmax = np.minimum(v3, v4), np.maximum(v3, v4)
            df[f"{name}_sum"] = v3 + v4
            df[f"{name}_absdiff"] = np.abs(v3 - v4)
            df[f"{name}_min"] = vmin
            df[f"{name}_max"] = vmax
            df[f"{name}_minmax_ratio"] = vmin / (vmax + 1e-8)
    return df

# ============================== PASS 1: read features ======================
print("Opening file (read pass) ...")
f_in = ROOT.TFile.Open(ROOT_FILE, "READ")
tree = f_in.Get(TREE_NAME)
n_entries = tree.GetEntries()
if MAX_EVENTS is not None:
    n_entries = min(n_entries, MAX_EVENTS)
print("  entries:", n_entries)

rows = []                 # feature dicts for VALID jets only
valid_flag = []           # per-entry list of bools (which jets are valid)
row_ptr = []              # per-entry list of indices into `rows` (or -1)

for i in range(n_entries):
    tree.GetEntry(i)
    n_jets = tree.jet_pt.size()
    entry_flags = []
    entry_ptrs = []
    for jet_i in range(n_jets):
        idx1 = int(tree.lund_max_kt_sd[jet_i])
        idx2 = int(tree.lund_max_kt_secondary_sd[jet_i])
        if idx1 < 0 or idx2 < 0:
            entry_flags.append(False)
            entry_ptrs.append(-1)
            continue
        try:
            row = {
                "z1":            tree.lund_z_sd[jet_i][idx1],
                "z2":            tree.lund_z_secondary_sd[jet_i][idx2],
                "deltaR34":      tree.lund_delta_secondary_sd[jet_i][idx2],
                "kt2":           tree.lund_kt_secondary_sd[jet_i][idx2],
                "N_charged3":    tree.lund_p3_n_charged[jet_i],
                "N_charged4":    tree.lund_p4_n_charged[jet_i],
                "N_all3":        tree.lund_p3_n_all[jet_i],
                "N_all4":        tree.lund_p4_n_all[jet_i],
                "pt_weight3":    tree.lund_p3_sigma[jet_i],
                "pt_weight4":    tree.lund_p4_sigma[jet_i],
                "pt_dispersion3":tree.lund_p3_ptD[jet_i],
                "pt_dispersion4":tree.lund_p4_ptD[jet_i],
            }
        except IndexError:
            entry_flags.append(False)
            entry_ptrs.append(-1)
            continue
        entry_ptrs.append(len(rows))
        entry_flags.append(True)
        rows.append(row)
    valid_flag.append(entry_flags)
    row_ptr.append(entry_ptrs)
    if i % 2000 == 0:
        print(f"  read {i}/{n_entries}")

f_in.Close()
print("  total valid jets:", len(rows))

# ============================== inference ==================================
if len(rows) == 0:
    raise RuntimeError("No valid jets found — check branch names / selection.")

df = pd.DataFrame(rows)

# stage-1: RAW features, pipeline transform, softmax col 1
X1 = df[s1_features].replace([np.inf, -np.inf], np.nan)
X1_pp = prep1.transform(X1).astype("float32")
pred1 = m1.predict(X1_pp, batch_size=BATCH, verbose=1)
matched_scores = (pred1[:, 1] if (pred1.ndim == 2 and pred1.shape[1] == 2)
                  else pred1.ravel()).astype(np.float64)

# stage-2: engineered features, dict transform, sigmoid
d2 = engineer_stage2(df)
missing = [c for c in feat2 if c not in d2.columns]
if missing:
    raise ValueError(f"stage-2 missing features: {missing}")
X2 = d2[feat2]
X2_pp = scl2.transform(imp2.transform(X2)).astype("float32")
qq_scores = m2.predict(X2_pp, batch_size=BATCH, verbose=1).ravel().astype(np.float64)

print("matched score range:", matched_scores.min().round(3), matched_scores.max().round(3))
print("qq score range:     ", qq_scores.min().round(3), qq_scores.max().round(3))

# ============================== PASS 2: write branches =====================
print("Opening file (update pass) ...")
f_out = ROOT.TFile.Open(ROOT_FILE, "UPDATE")
tree = f_out.Get(TREE_NAME)

# guard against re-running (duplicate branches)
for bname in ["lund_ML_qq", "lund_ML_matched"]:
    if tree.GetBranch(bname):
        print(f"  WARNING: branch {bname} already exists. Delete it first "
              f"(e.g. tree.SetBranchStatus) or work on a fresh copy. Aborting.")
        f_out.Close()
        raise SystemExit(1)

v_qq      = ROOT.std.vector("float")()
v_matched = ROOT.std.vector("float")()
b_qq      = tree.Branch("lund_ML_qq", v_qq)
b_matched = tree.Branch("lund_ML_matched", v_matched)

for i in range(n_entries):
    v_qq.clear(); v_matched.clear()
    for ptr in row_ptr[i]:
        if ptr < 0:
            v_qq.push_back(float(SENTINEL))
            v_matched.push_back(float(SENTINEL))
        else:
            v_qq.push_back(float(qq_scores[ptr]))
            v_matched.push_back(float(matched_scores[ptr]))
    b_qq.Fill()
    b_matched.Fill()
    if i % 2000 == 0:
        print(f"  wrote {i}/{n_entries}")

tree.Write("", ROOT.TObject.kOverwrite)
f_out.Close()
print("Done. Added branches lund_ML_qq and lund_ML_matched to", ROOT_FILE)