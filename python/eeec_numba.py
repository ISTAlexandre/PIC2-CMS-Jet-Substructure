import ROOT
import numpy as np
from numba import njit
import time


@njit
def fill_dpsi_hist_for_jet(pts, etas, phis, jet_pt, hist_counts,
                           nbins, xmin, xmax, symmetry_factor):
    """
    Computes one DeltaPsi per constituent triplet and fills a numpy histogram.

    hist_counts is modified in place.
    """

    n = len(pts)
    if n < 3 or jet_pt <= 0.0:
        return 0

    bin_width = (xmax - xmin) / nbins
    n_filled = 0

    # Build 3-vectors
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

    # Pairwise opening-angle matrix
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

    # Loop over unordered triplets
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
                    
                # Find closest pair. This defines thetaS and the pair used for DeltaPsi.
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

                # p_a, p_b, p_c
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

                # Normalize n1, n2, and u = (p_a + p_b)/|p_a + p_b|
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

                # Weight
                weight = symmetry_factor * pts[i] * pts[j] * pts[k] / (jet_pt ** 3)

                # Fill histogram manually
                if dpsi >= xmin and dpsi < xmax:
                    ibin = int((dpsi - xmin) / bin_width)
                    if ibin >= 0 and ibin < nbins:
                        hist_counts[ibin] += weight
                        n_filled += 1

    return n_filled


# -------------------------
# User settings
# -------------------------

file_path = "root/merged_pb.root"
tree_name = "jetTree"

hist_name = "lund_eeec_dpsi"

max_events = None          # None means run all events
max_const = None           # None means use all constituents. Example: 80 for testing.
qq_score_cut = -1            # Cut on the qq_score (0.0 to 1.0, -1 for no cut)
matched_score_cut = -1      # Cut on the matched_score (0.0 to 1.0, -1 for no cut)

nbins = 100
xmin = -np.pi
xmax = np.pi

# Use 1.0 to match your previous code.
# Use 8.0 if you want the explicit factor 8 from the paper convention.
symmetry_factor = 1.0


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

hist_counts = np.zeros(nbins, dtype=np.float64)

total_jets = 0
total_triplets_filled = 0

for iev in range(n_to_process):
    tree.GetEntry(iev)

    if iev % 1000 == 0:
        print(f"event {iev}/{n_to_process}")

    n_jets = len(tree.jet_pt)

    for jet_i in range(n_jets):
        jet_pt = float(tree.jet_pt[jet_i])
        qq_score = tree.lund_ML_qq[jet_i]
        matched_score = tree.lund_ML_matched[jet_i]
        # DNN CUT
        if qq_score < qq_score_cut or matched_score < matched_score_cut:
            continue

        pts = np.asarray(tree.const_pt[jet_i], dtype=np.float64)
        etas = np.asarray(tree.const_eta[jet_i], dtype=np.float64)
        phis = np.asarray(tree.const_phi[jet_i], dtype=np.float64)

        # Remove bad/zero-pt constituents
        mask = pts > 0.0
        pts = pts[mask]
        etas = etas[mask]
        phis = phis[mask]

        if len(pts) < 3:
            continue

        # Optional testing speed cut: keep only leading-pT constituents
        if max_const is not None and len(pts) > max_const:
            idx = np.argsort(pts)[-max_const:]
            pts = pts[idx]
            etas = etas[idx]
            phis = phis[idx]

        if len(pts) < 3:
            continue

        n_filled = fill_dpsi_hist_for_jet(
            pts,
            etas,
            phis,
            jet_pt,
            hist_counts,
            nbins,
            xmin,
            xmax,
            symmetry_factor,
        )

        total_jets += 1
        total_triplets_filled += n_filled

in_file.Close()

print(f"Processed jets: {total_jets}")
print(f"Filled triplets: {total_triplets_filled}")


# -------------------------
# Convert numpy histogram to ROOT TH1D
# -------------------------

hist = ROOT.TH1D(
    hist_name,
    "EEEC #Delta#psi;#Delta#psi;#Sigma",
    nbins,
    xmin,
    xmax,
)
hist.Sumw2()
hist.SetDirectory(0)

for ibin in range(nbins):
    # ROOT bins start at 1
    hist.SetBinContent(ibin + 1, float(hist_counts[ibin]))
    hist.SetBinError(ibin + 1, 0.0)


# -------------------------
# Save histogram into the existing ROOT file
# -------------------------

out_file = ROOT.TFile.Open(file_path, "UPDATE")
if not out_file or out_file.IsZombie():
    raise RuntimeError(f"Could not open {file_path} in UPDATE mode")

out_file.cd()

old_hist = out_file.Get(hist_name)
if old_hist:
    out_file.Delete(f"{hist_name};*")
    print(f"Deleted old histogram: {hist_name}")

hist.Write(hist_name, ROOT.TObject.kOverwrite)

out_file.Close()

end_time = time.time()

print(f"Saved histogram '{hist_name}' into {file_path}")
print(f"Processing completed in {round((end_time - start_time) / 60.0, 3)} minutes")