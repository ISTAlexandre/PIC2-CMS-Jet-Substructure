import ROOT
import numpy as np
from numba import njit
import time


@njit
def fill_dpsi_hist_for_jet(
    pts,
    etas,
    phis,
    jet_pt,
    hist_counts,
    nbins,
    xmin,
    xmax,
    symmetry_factor,
):
    """
    Computes one DeltaPsi per constituent triplet and fills a numpy histogram.

    hist_counts is modified in place.
    """

    n = len(pts)
    if n < 3 or jet_pt <= 0.0:
        return 0

    bin_width = (xmax - xmin) / nbins
    n_filled = 0

    pvecs = np.empty((n, 3), dtype=np.float64)
    norms = np.empty(n, dtype=np.float64)

    for i in range(n):
        pt = pts[i]
        eta = etas[i]
        phi = phis[i]

        px = pt * np.cos(phi)
        py = pt * np.sin(phi)
        pz = pt * np.sinh(eta)

        pvecs[i, 0] = px
        pvecs[i, 1] = py
        pvecs[i, 2] = pz

        norms[i] = np.sqrt(px * px + py * py + pz * pz)

    theta = np.empty((n, n), dtype=np.float64)

    for i in range(n):
        theta[i, i] = 0.0
        for j in range(i + 1, n):
            if norms[i] <= 0.0 or norms[j] <= 0.0:
                ang = np.nan
            else:
                dot = (
                    pvecs[i, 0] * pvecs[j, 0]
                    + pvecs[i, 1] * pvecs[j, 1]
                    + pvecs[i, 2] * pvecs[j, 2]
                )

                cosang = dot / (norms[i] * norms[j])

                if cosang > 1.0:
                    cosang = 1.0
                elif cosang < -1.0:
                    cosang = -1.0

                ang = np.arccos(cosang)

            theta[i, j] = ang
            theta[j, i] = ang

    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):

                theta_ij = theta[i, j]
                theta_ik = theta[i, k]
                theta_jk = theta[j, k]

                if not np.isfinite(theta_ij):
                    continue
                if not np.isfinite(theta_ik):
                    continue
                if not np.isfinite(theta_jk):
                    continue

                # ============ ADD THE CUT HERE ============
                # sort the three sides: R_S <= R_M <= R_L
                if theta_ij <= theta_ik:
                    if theta_ik <= theta_jk:
                        R_S, R_M, R_L = theta_ij, theta_ik, theta_jk
                    elif theta_ij <= theta_jk:
                        R_S, R_M, R_L = theta_ij, theta_jk, theta_ik
                    else:
                        R_S, R_M, R_L = theta_jk, theta_ij, theta_ik
                else:
                    if theta_ij <= theta_jk:
                        R_S, R_M, R_L = theta_ik, theta_ij, theta_jk
                    elif theta_ik <= theta_jk:
                        R_S, R_M, R_L = theta_ik, theta_jk, theta_ij
                    else:
                        R_S, R_M, R_L = theta_jk, theta_ik, theta_ij

                # Paper Eq. 16 cuts (absolute, radians):
                if not (np.sqrt(0.1) < R_L < 1.0):    # sqrt(0.1) < theta_L < 1
                    continue
                if not (0.01 < R_S < 0.1):          # 0.01 < theta_S < 0.1
                    continue

                # Closest pair defines the pair used for DeltaPsi
                if theta_ij <= theta_ik and theta_ij <= theta_jk:
                    a = i
                    b = j
                    c = k
                elif theta_ik <= theta_ij and theta_ik <= theta_jk:
                    a = i
                    b = k
                    c = j
                else:
                    a = j
                    b = k
                    c = i

                ax = pvecs[a, 0]
                ay = pvecs[a, 1]
                az = pvecs[a, 2]

                bx = pvecs[b, 0]
                by = pvecs[b, 1]
                bz = pvecs[b, 2]

                cx = pvecs[c, 0]
                cy = pvecs[c, 1]
                cz = pvecs[c, 2]

                # n1 = p_a x p_b
                n1x = ay * bz - az * by
                n1y = az * bx - ax * bz
                n1z = ax * by - ay * bx

                # p_ab = p_a + p_b
                abx = ax + bx
                aby = ay + by
                abz = az + bz

                # n2 = (p_a + p_b) x p_c
                n2x = aby * cz - abz * cy
                n2y = abz * cx - abx * cz
                n2z = abx * cy - aby * cx

                n1_norm = np.sqrt(n1x * n1x + n1y * n1y + n1z * n1z)
                n2_norm = np.sqrt(n2x * n2x + n2y * n2y + n2z * n2z)
                ab_norm = np.sqrt(abx * abx + aby * aby + abz * abz)

                if n1_norm <= 0.0 or n2_norm <= 0.0 or ab_norm <= 0.0:
                    continue

                n1x /= n1_norm
                n1y /= n1_norm
                n1z /= n1_norm

                n2x /= n2_norm
                n2y /= n2_norm
                n2z /= n2_norm

                ux = abx / ab_norm
                uy = aby / ab_norm
                uz = abz / ab_norm

                # cross(n1, n2)
                crossx = n1y * n2z - n1z * n2y
                crossy = n1z * n2x - n1x * n2z
                crossz = n1x * n2y - n1y * n2x

                numerator = ux * crossx + uy * crossy + uz * crossz
                denominator = n1x * n2x + n1y * n2y + n1z * n2z

                dpsi = np.arctan2(numerator, denominator)

                if not np.isfinite(dpsi):
                    continue

                weight = symmetry_factor * pts[i] * pts[j] * pts[k] / (jet_pt ** 3)

                if dpsi >= xmin and dpsi < xmax:
                    ibin = int((dpsi - xmin) / bin_width)
                    if ibin >= 0 and ibin < nbins:
                        hist_counts[ibin] += weight
                        n_filled += 1

    return n_filled


def make_root_hist(hist_name, hist_title, counts, nbins, xmin, xmax):
    hist = ROOT.TH1D(
        hist_name,
        hist_title,
        nbins,
        xmin,
        xmax,
    )
    hist.Sumw2()
    hist.SetDirectory(0)

    for ibin in range(nbins):
        hist.SetBinContent(ibin + 1, float(counts[ibin]))
        hist.SetBinError(ibin + 1, 0.0)

    return hist


# -------------------------
# User settings
# -------------------------

file_path = "root_ML/merged_ML-600to800.root"
tree_name = "jetTree"

max_events = None
max_const = None
qq_score_cut = -1 # Set to -1 to disable the qq_score cut
matched_score_cut = -1 # Set to -1 to disable the matched_score cut

nbins = 100
xmin = -np.pi
xmax = np.pi

# Use 1.0 to match your previous code.
# Use 8.0 if you want the explicit factor 8 convention.
symmetry_factor = 1.0

hist_all_name = "lund_eeec_dpsi"
hist_ch_names = [
    "lund_eeec_dpsi_ch0",
    "lund_eeec_dpsi_ch1",
    "lund_eeec_dpsi_ch2",
    "lund_eeec_dpsi_ch3",
    "lund_eeec_dpsi_ch4",
]


# -------------------------
# Main computation
# -------------------------

start_time = time.time()

in_file = ROOT.TFile.Open(file_path, "READ")
if not in_file or in_file.IsZombie():
    raise RuntimeError(f"Could not open {file_path}")

tree = in_file.Get(tree_name)
if not tree:
    raise RuntimeError(f"Could not find tree {tree_name}")

n_entries = tree.GetEntries()
print(f"Total entries in tree: {n_entries}")

if max_events is None:
    n_to_process = n_entries
else:
    n_to_process = min(max_events, n_entries)

# Combined histogram for labels 0 to 4 only
hist_all_counts = np.zeros(nbins, dtype=np.float64)

# Per-channel histograms: index 0 -> channel 0, ..., index 4 -> channel 4
hist_ch_counts = np.zeros((5, nbins), dtype=np.float64)

total_jets = 0
total_good_label_jets = 0
total_triplets_filled = 0
label_counts = np.zeros(6, dtype=np.int64)  # index 0 for -1, index 1..5 for 0..4

for iev in range(n_to_process):
    tree.GetEntry(iev)

    if iev % 1000 == 0:
        print(f"event {iev}/{n_to_process}")

    n_jets = len(tree.jet_pt)

    for jet_i in range(n_jets):
        total_jets += 1

        channel2 = int(tree.lund_secondary_idx_sd[jet_i])

        # Skip label -1 and anything outside 0..4
        if channel2 < 0 or channel2 > 4:
            if channel2 == -1:
                label_counts[0] += 1
            continue

        label_counts[channel2 + 1] += 1
        total_good_label_jets += 1

        jet_pt = float(tree.jet_pt[jet_i])
        qq_score = tree.lund_ML_qq[jet_i]
        matched_score = tree.lund_ML_matched[jet_i]

        #DNN CUT
        if qq_score < qq_score_cut or matched_score < matched_score_cut:
            continue

        pts = np.asarray(tree.const_pt[jet_i], dtype=np.float64)
        etas = np.asarray(tree.const_eta[jet_i], dtype=np.float64)
        phis = np.asarray(tree.const_phi[jet_i], dtype=np.float64)

        mask = pts > 0.0
        pts = pts[mask]
        etas = etas[mask]
        phis = phis[mask]

        if len(pts) < 3:
            continue

        if max_const is not None and len(pts) > max_const:
            idx = np.argsort(pts)[-max_const:]
            pts = pts[idx]
            etas = etas[idx]
            phis = phis[idx]

        if len(pts) < 3:
            continue

        # Fill combined histogram for all labels 0..4
        n_filled_all = fill_dpsi_hist_for_jet(
            pts,
            etas,
            phis,
            jet_pt,
            hist_all_counts,
            nbins,
            xmin,
            xmax,
            symmetry_factor,
        )

        # Fill the specific channel histogram
        n_filled_ch = fill_dpsi_hist_for_jet(
            pts,
            etas,
            phis,
            jet_pt,
            hist_ch_counts[channel2],
            nbins,
            xmin,
            xmax,
            symmetry_factor,
        )

        total_triplets_filled += n_filled_all

in_file.Close()

print(f"Processed jets total: {total_jets}")
print(f"Processed jets with channel2 in [0, 4]: {total_good_label_jets}")
print(f"Skipped jets with channel2 == -1: {label_counts[0]}")
print(f"channel2 == 0 jets: {label_counts[1]}")
print(f"channel2 == 1 jets: {label_counts[2]}")
print(f"channel2 == 2 jets: {label_counts[3]}")
print(f"channel2 == 3 jets: {label_counts[4]}")
print(f"channel2 == 4 jets: {label_counts[5]}")
print(f"Filled triplets in combined histogram: {total_triplets_filled}")


# -------------------------
# Convert numpy histograms to ROOT TH1D
# -------------------------

root_hists = []

hist_all = make_root_hist(
    hist_all_name,
    "EEEC #Delta#psi, channels 0-4;#Delta#psi;#Sigma",
    hist_all_counts,
    nbins,
    xmin,
    xmax,
)
root_hists.append(hist_all)

for ch in range(5):
    h = make_root_hist(
        hist_ch_names[ch],
        f"EEEC #Delta#psi, channel {ch};#Delta#psi;#Sigma",
        hist_ch_counts[ch],
        nbins,
        xmin,
        xmax,
    )
    root_hists.append(h)


# -------------------------
# Save histograms into existing ROOT file
# -------------------------

out_file = ROOT.TFile.Open(file_path, "UPDATE")
if not out_file or out_file.IsZombie():
    raise RuntimeError(f"Could not open {file_path} in UPDATE mode")

out_file.cd()

hist_names_to_delete = [hist_all_name] + hist_ch_names

for name in hist_names_to_delete:
    old = out_file.Get(name)
    if old:
        out_file.Delete(f"{name};*")
        print(f"Deleted old histogram: {name}")

for h in root_hists:
    h.Write(h.GetName(), ROOT.TObject.kOverwrite)
    print(f"Wrote histogram: {h.GetName()}")

out_file.Close()

end_time = time.time()

print(f"Saved EEEC histograms into {file_path}")
print(f"Processing completed in {round((end_time - start_time) / 60.0, 3)} minutes")