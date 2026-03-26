'''
mpiexec -n 6 python3 python/iterative_psi12.py
'''
import ROOT
from mpi4py import MPI
import time as time
import os
import shutil
import numpy as np

ROOT.gErrorIgnoreLevel = ROOT.kFatal
ROOT.gStyle.SetOptFit(1111)   # fit params + chi2/ndf, etc.
ROOT.gStyle.SetOptStat(0)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

local_folder = "root/"
local_files = [f for f in os.listdir(local_folder) if f.endswith(".root")]
local_files = local_files[rank::size]  # Distribute files among ranks
start_time = time.time()

pt_min = 200
eta_max = 1
n_imgs = 30

canvas = ROOT.TCanvas("canvas", "canvas", 800, 600)
psi12_hist = ROOT.TH1D("psi12_hist", "Psi12 Distribution;Psi12;Entries", 20, -np.pi, np.pi)
fit_curve = ROOT.TF1("fit_curve", "exp([0] + [1]*cos(2*x) + [2]*cos(3*x))", -np.pi, np.pi)
latex = ROOT.TLatex()
latex.SetNDC(True)
latex.SetTextSize(0.035)

for i, file in enumerate(local_files):
    print(f"Rank {rank} processing file {i+1}/{len(local_files)}: {file}")
    imgs_output_folder = f"iterative_psi12/{file[:-5]}"
 
    if os.path.isdir(imgs_output_folder):
        shutil.rmtree(imgs_output_folder)
    os.makedirs(imgs_output_folder, exist_ok=True)
    
    file = ROOT.TFile.Open(os.path.join(local_folder, file))
    tree = file.Get("jetTree")

    n_entries = tree.GetEntries()

    #Get tau_forms
    psi12_list = []
    tau_form_list = []
    for entry in range(n_entries):
        tree.GetEntry(entry)

        for n in range(len(tree.jet_pt)):
            pt = tree.jet_pt[n]
            eta = tree.jet_eta[n]

            if pt < pt_min or abs(eta) > eta_max or len(tree.tau_time[n]) < 1 or len(tree.lund_psi12_sd) < 1:
                continue
            
            tau_form = tree.tau_time[n][0]
            tau_form_list.append(tau_form)
            psi12 = tree.lund_psi12_sd[n]
            psi12_list.append(psi12)
    
    # Sort by tau_form
    if len(tau_form_list) == 0:
        print(f"Rank {rank}: no jets passed selection in {file.GetName()}, skipping.")
        continue
    order = np.argsort(np.asarray(tau_form_list, dtype=float))  # indices that sort tau
    tau_form_list = [tau_form_list[i] for i in order]
    psi12_list = [psi12_list[i] for i in order]

    # Fill histogram and save images
    n = len(tau_form_list)
    for p in range(n_imgs):
        start = int(p * n / n_imgs)
        end = int((p + 1) * n / n_imgs)

        psi12_hist.Reset()
        for idx in range(start, end):
            psi12_hist.Fill(psi12_list[idx])
        
        tau_lo = float(tau_form_list[start])
        tau_hi = float(tau_form_list[end-1])
        psi12_hist.SetTitle(f"Psi12 Distribution tau: {tau_lo:.3f} - {tau_hi:.3f};Psi12;Entries") 
        psi12_hist.Scale(1.0 / psi12_hist.Integral() if psi12_hist.Integral() > 0 else 1.0)  # Normalize

        canvas.cd()
        psi12_hist.Draw("hist")

        psi12_hist.Fit(fit_curve, "QR")
        fit_curve.SetLineColor(ROOT.kRed)
        fit_curve.Draw("same")

        subtitle = f"Jets with pt > {pt_min} GeV, |#eta| < {eta_max}"
        latex.DrawLatex(0.15, 0.85, subtitle)

        canvas.SaveAs(os.path.join(imgs_output_folder, f"dpsi12_{(p+1):03d}.png"))

    # Img of the total distribution
    psi12_hist.Reset()
    for idx in range(n):
        psi12_hist.Fill(psi12_list[idx])
    tau_lo = float(tau_form_list[0])
    tau_hi = float(tau_form_list[-1])
    psi12_hist.SetTitle(f"Psi12 Distribution tau: {tau_lo:.3f} - {tau_hi:.3f};Psi12;Entries")
    psi12_hist.Scale(1.0 / psi12_hist.Integral() if psi12_hist.Integral() > 0 else 1.0)  # Normalize
    canvas.cd()
    psi12_hist.Draw("hist")
    subtitle = f"Jets with pt > {pt_min} GeV, |#eta| < {eta_max}"
    latex.DrawLatex(0.15, 0.85, subtitle)
    canvas.SaveAs(f"{imgs_output_folder}/psi12_total.png")

end_time = time.time()
print(f"Rank {rank} finished processing {len(local_files)} files in {end_time - start_time:.2f} seconds.")

