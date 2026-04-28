'''
mpiexec -n 6 python3 python/iterative_psi_z.py
'''
import ROOT
from mpi4py import MPI
import time as time
import os
import shutil
import numpy as np


ROOT.gErrorIgnoreLevel = ROOT.kFatal
ROOT.gStyle.SetOptStat(0)
ROOT.gROOT.SetBatch(True)

def ensure_dir(path):
    # Remove if it exists, then recreate
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

def plot_save_hist(hist, canvas, tau_lo, tau_hi, output_path, pt_min, eta_max, p=None, title="fig"):
    # clear canvas
    canvas.Clear()

    hist.SetTitle(f"{title}_1 vs {title}_2 for z1: [{tau_lo:.3f}, {tau_hi:.3f}]")
    integral = hist.Integral()
    if integral <= 0:
        print(f"Rank {rank}: z1 range [{tau_lo:.3f}, {tau_hi:.3f}] has {integral} entries, skipping.")
        return False

    #hist.Scale(1.0 / integral)
    canvas.cd()
    hist.Draw("COLZ")
    subtitle = f"Jets with p_{{T}} > {pt_min:g} GeV/c and |#eta| < {eta_max:g}"
    latex.DrawLatex(0.18, 0.92, subtitle)

    canvas.Update()

    # filename: either indexed (frames) or total
    fname = f"{title}_{(p+1):03d}.png" if p is not None else f"{title}_total.png"
    canvas.SaveAs(os.path.join(output_path, fname))
    return True

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

local_folder = "root/"
local_files = [f for f in os.listdir(local_folder) if f.endswith(".root")]
local_files = local_files[rank::size]  # Distribute files among ranks
start_time = time.time()

pt_min = 600
eta_max = 1.7
n_imgs = 15

canvas = ROOT.TCanvas("canvas", "canvas", 1000, 800)
psi12_hist = ROOT.TH2D("psi12_hist", "psi12 distribution;psi12;z2", 20, 0, np.pi, 10, 0.1, 0.5)

latex = ROOT.TLatex()
latex.SetNDC(True)
latex.SetTextSize(0.035)

for i, file in enumerate(local_files):
    print(f"Rank {rank} processing file {i+1}/{len(local_files)}: {file}")
    psi12_output_folder = f"iterative_psi12_z/{file[:-5]}"

    if "pbpb" in os.path.basename(file).lower():
        pt_min = 10
        print(f"Rank {rank}: identified PbPb file, setting pt_min to {pt_min} GeV/c.")
    else:
        pt_min = 600

    ensure_dir(psi12_output_folder)

    file = ROOT.TFile.Open(os.path.join(local_folder, file))
    tree = file.Get("jetTree")

    n_entries = tree.GetEntries()
    psi12_list = []
    z1_list = []
    z2_list = []

    for entry in range(n_entries):
        tree.GetEntry(entry)

        for n in range(len(tree.jet_pt)):
            if tree.jet_pt[n] < pt_min or abs(tree.jet_eta[n]) > eta_max or len(tree.lund_z_sd[n]) < 1 or len(tree.lund_z_secondary_sd[n]) < 1:
                continue

            max_kt_index1 = int(tree.lund_max_kt_sd[n])
            max_kt_index2 = int(tree.lund_max_kt_secondary_sd[n])

            if max_kt_index1 < 0 or max_kt_index2 < 0:
                continue

            z1 = tree.lund_z_sd[n][max_kt_index1]
            z2 = tree.lund_z_secondary_sd[n][max_kt_index2]
            psi12 = tree.lund_psi12_sd[n]

            z1_list.append(z1)
            z2_list.append(z2)
            psi12_list.append(psi12)

    order = np.argsort(np.asarray(z1_list))
    z1_list = [z1_list[i] for i in order]
    z2_list = [z2_list[i] for i in order]
    psi12_list = [psi12_list[i] for i in order]
    n = len(z1_list)
    for p in range(n_imgs):
        start = int(p * n / n_imgs)
        end = int((p + 1) * n / n_imgs)

        psi12_hist.Reset()
        for r in range(start, end):
            psi12_hist.Fill(psi12_list[r], z2_list[r])

        z1_lo = z1_list[start]
        z1_hi = z1_list[end-1]
        plot_save_hist(psi12_hist, canvas, z1_lo, z1_hi, psi12_output_folder, pt_min, eta_max, p=p,title="psi12")

    # Save total histogram
    psi12_hist.Reset()
    for p in range(n):
        psi12_hist.Fill(psi12_list[p], z2_list[p])
    plot_save_hist(psi12_hist, canvas, z1_list[0], z1_list[-1], psi12_output_folder, pt_min, eta_max, p=None,title="psi12")

end_time = time.time()
print(f"Rank {rank} finished processing in {end_time - start_time:.2f} seconds.")


