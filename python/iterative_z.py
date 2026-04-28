'''
mpiexec -n 6 python3 python/iterative_z.py
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

    hist.SetTitle(f"{title}_1 vs {title}_2 for tau: [{tau_lo:.3f}, {tau_hi:.3f}]")
    integral = hist.Integral()
    if integral <= 0:
        print(f"Rank {rank}: tau range [{tau_lo:.3f}, {tau_hi:.3f}] has {integral} entries, skipping.")
        return False

    hist.Scale(1.0 / integral)
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
z_hist = ROOT.TH2D("z_hist", "z_1 vs z_2;z_1;z_2", 50, 0, 1, 50, 0, 1)
kt_hist = ROOT.TH2D("kt_hist", "kT_1 vs kT_2;kT_1;kT_2", 100, 0, 1, 100, 0, 1)
delta_hist = ROOT.TH2D("delta_hist", "Delta_1 vs Delta_2;Delta_1;Delta_2", 100, 0, 1, 100, 0, 1)
psi12_hist = ROOT.TH1D("psi12_hist", "Psi12 Distribution;Psi12;Entries", 25, 0, np.pi)

latex = ROOT.TLatex()
latex.SetNDC(True)
latex.SetTextSize(0.035)

for i, file in enumerate(local_files):
    print(f"Rank {rank} processing file {i+1}/{len(local_files)}: {file}")
    imgs_output_folder = f"iterative_z/{file[:-5]}"
    kt_output_folder = f"iterative_kt/{file[:-5]}"
    delta_output_folder = f"iterative_delta/{file[:-5]}"
    psi12_output_folder = f"iterative_psi12/{file[:-5]}"

    if "pbpb" in os.path.basename(file).lower():
        pt_min = 10
        print(f"Rank {rank}: identified PbPb file, setting pt_min to {pt_min} GeV/c.")
    else:
        pt_min = 600
    
    ensure_dir(imgs_output_folder)
    ensure_dir(kt_output_folder)
    ensure_dir(delta_output_folder)
    ensure_dir(psi12_output_folder)

    file = ROOT.TFile.Open(os.path.join(local_folder, file))
    tree = file.Get("jetTree")

    n_entries = tree.GetEntries()

    #Get tau_forms
    tau_form_list = []
    z1_list = []
    z2_list = []
    kt1_list = []
    kt2_list = []
    delta1_list = []
    delta2_list = []
    psi12_list = []
    for entry in range(n_entries):
        tree.GetEntry(entry)

        for n in range(len(tree.jet_pt)):
            pt = tree.jet_pt[n]
            eta = tree.jet_eta[n]

            if pt < pt_min or abs(eta) > eta_max or len(tree.tau_time[n]) < 1 or len(tree.lund_z_sd[n]) < 1 or len(tree.lund_z_secondary_sd[n]) < 1:
                continue

            max_kt_index1 = int(tree.lund_max_kt_sd[n])
            max_kt_index2 = int(tree.lund_max_kt_secondary_sd[n])

            if max_kt_index1 < 0 or max_kt_index2 < 0:
                continue

            tau_form = tree.tau_time[n][0]
            tau_form_list.append(tau_form)

            z1_list.append(tree.lund_z_sd[n][max_kt_index1])
            z2_list.append(tree.lund_z_secondary_sd[n][max_kt_index2])

            kt1_list.append(tree.lund_kt_sd[n][max_kt_index1])
            kt2_list.append(tree.lund_kt_secondary_sd[n][max_kt_index2])

            delta1_list.append(tree.lund_delta_sd[n][max_kt_index1])
            delta2_list.append(tree.lund_delta_secondary_sd[n][max_kt_index2])

            psi12_list.append(tree.lund_psi12_sd[n])

    
    #Sort by tau_form
    order = np.argsort(np.asarray(tau_form_list, dtype=float))  # indices that sort tau
    tau_form_list = [tau_form_list[i] for i in order]
    z1_list = [z1_list[i] for i in order]
    z2_list = [z2_list[i] for i in order]
    kt1_list = [kt1_list[i] for i in order]
    kt2_list = [kt2_list[i] for i in order]
    delta1_list = [delta1_list[i] for i in order]
    delta2_list = [delta2_list[i] for i in order]
    psi12_list = [psi12_list[i] for i in order]

    z_hist.GetXaxis().SetRangeUser(min(z1_list), max(z1_list))
    z_hist.GetYaxis().SetRangeUser(min(z2_list), max(z2_list))

    kt_hist.GetXaxis().SetRangeUser(0.95, max(kt1_list))
    kt_hist.GetYaxis().SetRangeUser(min(kt2_list), max(kt2_list))

    delta_hist.GetXaxis().SetRangeUser(min(delta1_list), max(delta1_list))
    delta_hist.GetYaxis().SetRangeUser(min(delta2_list), max(delta2_list))

    n = len(tau_form_list)
    for p in range(n_imgs):
        start = (p * n) // n_imgs
        end = ((p + 1) * n) // n_imgs

        z_hist.Reset()
        kt_hist.Reset()
        delta_hist.Reset()
        psi12_hist.Reset()
        for i in range(start, end):
            z_hist.Fill(z1_list[i], z2_list[i])
            kt_hist.Fill(kt1_list[i], kt2_list[i])
            delta_hist.Fill(delta1_list[i], delta2_list[i])
            psi12_hist.Fill(psi12_list[i])
        
        tau_lo = tau_form_list[start]
        tau_hi = tau_form_list[end-1]

        plot_save_hist(z_hist, canvas, tau_lo, tau_hi, imgs_output_folder, pt_min, eta_max, p=p, title="z")
        plot_save_hist(kt_hist, canvas, tau_lo, tau_hi, kt_output_folder, pt_min, eta_max, p=p, title="kt")
        plot_save_hist(delta_hist, canvas, tau_lo, tau_hi, delta_output_folder, pt_min, eta_max, p=p,title="delta")

        canvas.Clear()
        psi12_hist.SetTitle(f"psi12 distribution for tau: [{tau_lo:.3f}, {tau_hi:.3f}]")
        integral = psi12_hist.Integral()
        if integral > 0:
            psi12_hist.Scale(1.0 / integral)
            psi12_hist.Draw("hist")
            subtitle = f"Jets with p_{{T}} > {pt_min:g} GeV/c and |#eta| < {eta_max:g}"
            latex.DrawLatex(0.18, 0.92, subtitle)
            canvas.Update()
            fname = f"psi12_{(p+1):03d}.png"
            canvas.SaveAs(os.path.join(psi12_output_folder, fname))
    
    z_hist.Reset()
    kt_hist.Reset()
    delta_hist.Reset()
    for i in range(n):
        z_hist.Fill(z1_list[i], z2_list[i])
        kt_hist.Fill(kt1_list[i], kt2_list[i])
        delta_hist.Fill(delta1_list[i], delta2_list[i])
        psi12_hist.Fill(psi12_list[i])
    plot_save_hist(z_hist, canvas, tau_form_list[0], tau_form_list[-1], imgs_output_folder, pt_min, eta_max, p=None, title="z")
    plot_save_hist(kt_hist, canvas, tau_form_list[0], tau_form_list[-1], kt_output_folder, pt_min, eta_max, p=None, title="kt")
    plot_save_hist(delta_hist, canvas, tau_form_list[0], tau_form_list[-1], delta_output_folder, pt_min, eta_max, p=None,title="delta")

    canvas.Clear()
    psi12_hist.SetTitle(f"psi12 distribution for tau: [{tau_form_list[0]:.3f}, {tau_form_list[-1]:.3f}]")
    integral = psi12_hist.Integral()
    if integral > 0:
        psi12_hist.Scale(1.0 / integral)
        psi12_hist.Draw("hist")
        subtitle = f"Jets with p_{{T}} > {pt_min:g} GeV/c and |#eta| < {eta_max:g}"
        latex.DrawLatex(0.18, 0.92, subtitle)
        canvas.Update()
        fname = f"psi12_total.png"
        canvas.SaveAs(os.path.join(psi12_output_folder, fname))

end_time = time.time()
print(f"Rank {rank} finished processing in {end_time - start_time:.2f} seconds.")