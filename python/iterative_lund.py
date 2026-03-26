'''
mpiexec -n 6 python3 python/iterative_lund.py
'''
import ROOT
from mpi4py import MPI
import time as time
import os
import shutil
import numpy as np

ROOT.gErrorIgnoreLevel = ROOT.kFatal

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
lund_hist = ROOT.TH2D("lund_hist", "Lund Plane;ln(1/Delta);ln(kT)", 100, 0.5, 8, 100, -7, 6)
latex = ROOT.TLatex()
latex.SetNDC(True)
latex.SetTextSize(0.035)

for i, file in enumerate(local_files):
    print(f"Rank {rank} processing file {i+1}/{len(local_files)}: {file}")
    imgs_output_folder = f"iterative_lund/{file[:-5]}"

    if os.path.isdir(imgs_output_folder):
        shutil.rmtree(imgs_output_folder)
    os.makedirs(imgs_output_folder, exist_ok=True)
    
    file = ROOT.TFile.Open(os.path.join(local_folder, file))
    tree = file.Get("jetTree")

    n_entries = tree.GetEntries()

    #Get tau_forms
    tau_form_list = []
    x_list = []
    y_list = []
    for entry in range(n_entries):
        tree.GetEntry(entry)

        for n in range(len(tree.jet_pt)):
            pt = tree.jet_pt[n]
            eta = tree.jet_eta[n]

            if pt < pt_min or abs(eta) > eta_max or len(tree.tau_time[n]) < 1 or len(tree.lund_coords_x_sd[n]) < 1:
                continue
            
            tau_form = tree.tau_time[n][0]
            tau_form_list.append(tau_form)
            x_sublist = []
            y_sublist = []
            for lund_i in range(len(tree.lund_coords_x_sd[n])):
                x_sublist.append(tree.lund_coords_x_sd[n][lund_i])
                y_sublist.append(tree.lund_coords_y_sd[n][lund_i])
            x_list.append(x_sublist)
            y_list.append(y_sublist)

    # Sort by tau (min -> max) and apply the same ordering to x_list and y_list
    if len(tau_form_list) == 0:
        print(f"Rank {rank}: no jets passed selection in {file.GetName()}, skipping.")
        continue

    order = np.argsort(np.asarray(tau_form_list, dtype=float))  # indices that sort tau
    tau_form_list = [tau_form_list[i] for i in order]
    x_list = [x_list[i] for i in order]
    y_list = [y_list[i] for i in order]

    n = len(tau_form_list)
    for p in range(n_imgs):
        start = (p * n) // n_imgs
        end = ((p + 1) * n) // n_imgs

        lund_hist.Reset()
        for idx in range(start, end):
            for j in range(len(x_list[idx])):
                lund_hist.Fill(x_list[idx][j], y_list[idx][j])

        tau_lo = float(tau_form_list[start])
        tau_hi = float(tau_form_list[end - 1])
        lund_hist.SetTitle(f"Lund Plane ({tau_lo:.3f} < tau < {tau_hi:.3f})")
        integral = lund_hist.Integral()
        if integral <= 0:
            print(f"file {file.GetName()}, tau range [{tau_lo:.3f}, {tau_hi:.3f}] has zero entries, skipping.")
            continue
        lund_hist.Scale(1/integral)  # Normalize to unit area
        canvas.cd()
        lund_hist.Draw("COLZ")

        subtitle = f"Jets with p_{{T}} > {pt_min:g} GeV/c and |#eta| < {eta_max:g}"
        latex.DrawLatex(0.18, 0.92, subtitle)  # (x,y) in NDC

        canvas.SaveAs(os.path.join(imgs_output_folder, f"lund_{(p+1):03d}.png"))
    
    #img of the total lund plane
    lund_hist.Reset()
    for idx in range(n):
        for j in range(len(x_list[idx])):
            lund_hist.Fill(x_list[idx][j], y_list[idx][j])
    tau_lo = float(tau_form_list[0])
    tau_hi = float(tau_form_list[-1])
    lund_hist.SetTitle(f"Lund Plane ({tau_lo:.3f} < #tau < {tau_hi:.3f})")
    integral = lund_hist.Integral()
    if integral > 0:
        lund_hist.Scale(1/integral)  # Normalize to unit area
        canvas.cd()
        lund_hist.Draw("COLZ")
        subtitle = f"Jets with p_{{T}} > {pt_min:g} GeV/c and |#eta| < {eta_max:g}"
        latex.DrawLatex(0.18, 0.92, subtitle)  # (x,y) in NDC
        canvas.SaveAs(os.path.join(imgs_output_folder, f"lund_total.png"))
    


end_time = time.time()
print(f"Rank {rank} finished processing in {end_time - start_time:.2f} seconds.")
