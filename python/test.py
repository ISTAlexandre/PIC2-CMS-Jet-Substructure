import ROOT
import numpy as np
import matplotlib.pyplot as plt

def wrap_to_pi(angle):
    """Wrap angle to the range [-pi, pi]."""
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle

file_path = "root_ML/merged700.root"
#file_path = "root/out_pbpb.root"

file = ROOT.TFile.Open(file_path)
tree = file.Get("jetTree")

n_entries = tree.GetEntries()
print(f"Number of entries in the tree: {n_entries}")

max_pt = 700
eta_lim = 1.7

hist = ROOT.TH1F("hist", "dpsi12 (channel 1)",10, -0,np.pi) #qq
hist.SetLineColor(ROOT.kBlue)
hist.Sumw2()

hist2 = ROOT.TH1F("hist2", "dpsi12 (channel 2)",10, -0,np.pi) #qg
hist2.SetLineColor(ROOT.kGreen)
hist2.Sumw2()

hist3 = ROOT.TH1F("hist3", "dpsi12 (channel 3)",10, -0,np.pi) #gg
hist3.SetLineColor(ROOT.kRed)
hist3.Sumw2()

hist4 = ROOT.TH1F("hist4", "dpsi12 (channel 4)",10, -0,np.pi) #rest
hist4.SetLineColor(ROOT.kMagenta)
hist4.Sumw2()

hist0 = ROOT.TH1F("hist0", "dpsi12 (channel 0)",10, -0,np.pi) #unmatched
hist0.SetLineColor(ROOT.kBlack)
hist0.Sumw2()

histz = ROOT.TH2F("histz", "z1 vs z2", 20, 0, 0.5, 20, 0, 0.5)

hist_lund1 = ROOT.TH2F("hist_lund1", "Lund Plane 1;ln(1/Delta);ln(kT)", 100, 0.5, 8, 100, -2.5, 6)
hist_lund2 = ROOT.TH2F("hist_lund2", "Lund Plane 2;ln(1/Delta);ln(kT)", 100, 0.5, 8, 100, -2.5, 6)

hist_channel1 = ROOT.TH1F("hist_channel1", "Channel 1", 5, -1, 4)
hist_channel2 = ROOT.TH1F("hist_channel2", "Channel 2", 5, -1, 4)

fit_func = ROOT.TF1("fit_func", "e**([0]+[1]*cos(2*x))", -0, np.pi)
fit_func.SetLineColor(ROOT.kRed)
#+[2]*cos(4*x)
fit_func2 = ROOT.TF1("fit_func2", "[0]+[1]*cos(2*x)", -0, np.pi)
fit_func2.SetLineColor(ROOT.kGreen)

canvas = ROOT.TCanvas("canvas", "Canvas", 800, 600)

hist_pt2_channel1 = ROOT.TH1F("hist_pt2_channel1", "pT2 (channel 1)", 50, 0, 400) #qq

sum_c2 = 0
sum_c2_sq = 0
j = 0
for i in range(n_entries):
    tree.GetEntry(i)
    for n in range(len(tree.jet_pt)):
        if tree.jet_pt[n] < max_pt or abs(tree.jet_eta[n]) > eta_lim:# or len(tree.tau_time[n]) < 1:
            continue
        
        max_kt1 = tree.lund_max_kt_sd[n]
        max_kt2 = tree.lund_max_kt_secondary_sd[n]

        if max_kt1 < 0 or max_kt2 < 0:
            continue

        z1 = tree.lund_z_sd[n][max_kt1]
        z2 = tree.lund_z_secondary_sd[n][max_kt2]

        x1 = tree.lund_coords_x_sd[n][max_kt1]
        y1 = tree.lund_coords_y_sd[n][max_kt1]
        hist_lund1.Fill(x1, y1)

        x2 = tree.lund_coords_secondary_x_sd[n][max_kt2]
        y2 = tree.lund_coords_secondary_y_sd[n][max_kt2]
        hist_lund2.Fill(x2, y2)

        channel1 = tree.lund_primary_idx_sd[n]
        channel2 = tree.lund_secondary_idx_sd[n]
        hist_channel1.Fill(channel1)
        hist_channel2.Fill(channel2)

        dpsi12 = tree.lund_psi12_sd[n]
    
        if channel2 == 1: #qq
            hist.Fill(dpsi12)
            hist_pt2_channel1.Fill(tree.lund_max_kt_pt2_sd[n])
        
        if channel2 == 2: #qg
            hist2.Fill(dpsi12)
        
        if channel2 == 3: #gg
            hist3.Fill(dpsi12)

        if channel2 ==4: #rest
            hist4.Fill(dpsi12)

        if channel2 == 0: #unmatched
            hist0.Fill(dpsi12)
       
        value = np.cos(2*dpsi12)
        sum_c2 += value
        sum_c2_sq += value**2
        j += 1

        histz.Fill(z1, z2)

fit_func2.SetParameter(0, (hist.GetMaximum()-hist.GetMinimum())/2)
fit_func2.SetParLimits(1, -hist.GetMaximum(), hist.GetMaximum())

hist.Fit(fit_func, "IEWRMG")
hist.Fit(fit_func2, "RME")
hist.Draw("HIST1")
fit_func.Draw("same")
fit_func2.Draw("same")

total_int = hist.Integral() + hist2.Integral() + hist3.Integral() + hist4.Integral() + hist0.Integral()
percentage0 = hist0.Integral() / total_int * 100
percentage1 = hist.Integral() / total_int * 100
percentage2 = hist2.Integral() / total_int * 100
percentage3 = hist3.Integral() / total_int * 100
percentage4 = hist4.Integral() / total_int * 100

print(f"Channel 0 (unmatched): {percentage0:.2f}% / 72% - {hist0.Integral()}")
print(f"Channel 1 (qq): {percentage1:.2f}% / 3% - {hist.Integral()}")
print(f"Channel 2 (qg): {percentage2:.2f}% / 8% - {hist2.Integral()}")
print(f"Channel 3 (gg): {percentage3:.2f}% / 16% - {hist3.Integral()}")
print(f"Channel 4 (rest): {percentage4:.2f}% / 1% - {hist4.Integral()}")

canvas.Draw()
canvas.Update()
canvas.WaitPrimitive()

histz.Draw("COLZ")
canvas.Draw()
canvas.Update()
canvas.WaitPrimitive()

hist_lund1.Draw("COLZ")
canvas.Draw()
canvas.Update()
canvas.WaitPrimitive()

hist_lund2.Draw("COLZ")
canvas.Draw()
canvas.Update()
canvas.WaitPrimitive()

hist_channel1.Draw("HIST")
canvas.Draw()
canvas.Update()
canvas.WaitPrimitive()

hist_channel2.Draw("HIST")
canvas.Draw()
canvas.Update()
canvas.WaitPrimitive()

hist_pt2_channel1.Draw("HIST")
canvas.Draw()
canvas.Update()
canvas.WaitPrimitive()

if j == 0:
    print("No valid entries found for fitting.")
else:
    mean_c2 = sum_c2 / j

    var = (sum_c2_sq / j) - (mean_c2 ** 2)
    err = np.sqrt(max(var,0) / j)
    
    print("N = ", j)
    print("<cos(2 psi12)> = {:.6f} ± {:.6f}".format(mean_c2, err))
