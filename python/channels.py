import ROOT
import numpy as np

ROOT.gStyle.SetOptStat(0)

def pt1_calc(pt2,z):
    return (pt2*(1-z))/z


file_path = "root_ML/merged700.root"

file = ROOT.TFile.Open(file_path)
tree = file.Get("jetTree")

n_entries = tree.GetEntries()
print(f"Number of entries in the tree: {n_entries}")

max_pt = 700
eta_lim = 1.7


hs = ROOT.THStack("hs", "Stacked Histograms fot pt2")

hist_pt2_channel1 = ROOT.TH1F("hist_pt2_channel1", "lund_pt2 (channel 1)", 50, 0, 400)
hist_pt2_channel1.SetLineColor(ROOT.kBlue)
hist_pt2_channel1.SetFillColor(ROOT.kBlue)
hist_pt2_channel1.Sumw2()
hs.Add(hist_pt2_channel1)

hist_pt2_channel2 = ROOT.TH1F("hist_pt2_channel2", "lund_pt2 (channel 2)", 50, 0, 400)
hist_pt2_channel2.SetLineColor(ROOT.kGreen)
hist_pt2_channel2.SetFillColor(ROOT.kGreen)
hist_pt2_channel2.Sumw2()
hs.Add(hist_pt2_channel2)

hist_pt2_channel3 = ROOT.TH1F("hist_pt2_channel3", "lund_pt2 (channel 3)", 50, 0, 400)
hist_pt2_channel3.SetLineColor(ROOT.kRed)
hist_pt2_channel3.SetFillColor(ROOT.kRed)
hist_pt2_channel3.Sumw2()
hs.Add(hist_pt2_channel3)

hs_psi12 = ROOT.THStack("hs_psi12", "Stacked Histograms for psi12")

hist_psi12_channel1 = ROOT.TH1F("hist_psi12_channel1", "lund_psi12 (channel 1)", 20, 0, np.pi)
hist_psi12_channel1.SetLineColor(ROOT.kBlue)
hist_psi12_channel1.SetFillColor(ROOT.kBlue)
hist_psi12_channel1.Sumw2()
hs_psi12.Add(hist_psi12_channel1)

hist_psi12_channel2 = ROOT.TH1F("hist_psi12_channel2", "lund_psi12 (channel 2)", 20, 0, np.pi)
hist_psi12_channel2.SetLineColor(ROOT.kGreen)
hist_psi12_channel2.SetFillColor(ROOT.kGreen)
hist_psi12_channel2.Sumw2()
hs_psi12.Add(hist_psi12_channel2)

hist_psi12_channel3 = ROOT.TH1F("hist_psi12_channel3", "lund_psi12 (channel 3)", 20, 0, np.pi)
hist_psi12_channel3.SetLineColor(ROOT.kRed)
hist_psi12_channel3.SetFillColor(ROOT.kRed)
hist_psi12_channel3.Sumw2()
hs_psi12.Add(hist_psi12_channel3)

hist_z_channel1 = ROOT.TH2F("hist_z_channel1", "z1 vs z2 (channel 1)", 20, 0, 0.5, 20, 0, 0.5)

hist_z_channel2 = ROOT.TH2F("hist_z_channel2", "z1 vs z2 (channel 2)", 20, 0, 0.5, 20, 0, 0.5)

hist_z_channel3 = ROOT.TH2F("hist_z_channel3", "z1 vs z2 (channel 3)", 20, 0, 0.5, 20, 0, 0.5)

hs_pt1 = ROOT.THStack("hs_pt1", "Stacked Histograms for pt1")

hist_pt1_channel1 = ROOT.TH1F("hist_pt1_channel1", "lund_pt1 (channel 1)", 50, 0, 500)
hist_pt1_channel1.SetLineColor(ROOT.kBlue)
hist_pt1_channel1.SetFillColor(ROOT.kBlue)
hist_pt1_channel1.Sumw2()
hs_pt1.Add(hist_pt1_channel1)

hist_pt1_channel2 = ROOT.TH1F("hist_pt1_channel2", "lund_pt1 (channel 2)", 50, 0, 500)
hist_pt1_channel2.SetLineColor(ROOT.kGreen)
hist_pt1_channel2.SetFillColor(ROOT.kGreen)
hist_pt1_channel2.Sumw2()
hs_pt1.Add(hist_pt1_channel2)

hist_pt1_channel3 = ROOT.TH1F("hist_pt1_channel3", "lund_pt1 (channel 3)", 50, 0, 500)
hist_pt1_channel3.SetLineColor(ROOT.kRed)
hist_pt1_channel3.SetFillColor(ROOT.kRed)
hist_pt1_channel3.Sumw2()
hs_pt1.Add(hist_pt1_channel3)

hs_tau_time = ROOT.THStack("hs_tau_time", "Stacked Histograms for tau_time")

hist_tau_time_channel1 = ROOT.TH1F("hist_tau_time_channel1", "tau_time (channel 1)", 50, 0, 0.5)
hist_tau_time_channel1.SetLineColor(ROOT.kBlue)
hist_tau_time_channel1.SetFillColor(ROOT.kBlue)
hist_tau_time_channel1.Sumw2()
hs_tau_time.Add(hist_tau_time_channel1)

hist_tau_time_channel2 = ROOT.TH1F("hist_tau_time_channel2", "tau_time (channel 2)", 50, 0, 0.5)
hist_tau_time_channel2.SetLineColor(ROOT.kGreen)
hist_tau_time_channel2.SetFillColor(ROOT.kGreen)
hist_tau_time_channel2.Sumw2()
hs_tau_time.Add(hist_tau_time_channel2)

hist_tau_time_channel3 = ROOT.TH1F("hist_tau_time_channel3", "tau_time (channel 3)", 50, 0, 0.5)
hist_tau_time_channel3.SetLineColor(ROOT.kRed)
hist_tau_time_channel3.SetFillColor(ROOT.kRed)
hist_tau_time_channel3.Sumw2()
hs_tau_time.Add(hist_tau_time_channel3)


for n in range(n_entries):
    tree.GetEntry(n)
    
    for i in range(len(tree.jet_pt)):

        channel2 = tree.lund_secondary_idx_sd[i]
        max_kt1 = tree.lund_max_kt_sd[i]
        max_kt2 = tree.lund_max_kt_secondary_sd[i]

        has_tau_time = False
        if len(tree.tau_time[i]) > 0:
            tau_time = tree.tau_time[i][0]
            has_tau_time = True
        
        if channel2 == 1: #qq
            hist_pt2_channel1.Fill(tree.lund_max_kt_pt2_sd[i])
            hist_psi12_channel1.Fill(tree.lund_psi12_sd[i])
            hist_z_channel1.Fill(tree.lund_z_sd[i][max_kt1], tree.lund_z_secondary_sd[i][max_kt2])
            hist_pt1_channel1.Fill(pt1_calc(tree.lund_max_kt_pt2_sd[i], tree.lund_z_sd[i][max_kt1]))
            if has_tau_time: hist_tau_time_channel1.Fill(tau_time)
        
        if channel2 == 2: #qg
            hist_pt2_channel2.Fill(tree.lund_max_kt_pt2_sd[i])
            hist_psi12_channel2.Fill(tree.lund_psi12_sd[i])
            hist_z_channel2.Fill(tree.lund_z_sd[i][max_kt1], tree.lund_z_secondary_sd[i][max_kt2])
            hist_pt1_channel2.Fill(pt1_calc(tree.lund_max_kt_pt2_sd[i], tree.lund_z_sd[i][max_kt1]))
            if has_tau_time: hist_tau_time_channel2.Fill(tau_time)
        
        if channel2 == 3: #gg
            hist_pt2_channel3.Fill(tree.lund_max_kt_pt2_sd[i])
            hist_psi12_channel3.Fill(tree.lund_psi12_sd[i])
            hist_z_channel3.Fill(tree.lund_z_sd[i][max_kt1], tree.lund_z_secondary_sd[i][max_kt2])
            hist_pt1_channel3.Fill(pt1_calc(tree.lund_max_kt_pt2_sd[i], tree.lund_z_sd[i][max_kt1]))
            if has_tau_time: hist_tau_time_channel3.Fill(tau_time)

canvas = ROOT.TCanvas("canvas", "Canvas", 800, 600)
hs.Draw("HIST")

canvas.cd()

leg = ROOT.TLegend(0.65, 0.70, 0.88, 0.88)  # (x1,y1,x2,y2) in NDC
leg.SetBorderSize(0)
leg.SetFillStyle(0)
leg.AddEntry(hist_pt2_channel1, "qq (channel 1)", "f")
leg.AddEntry(hist_pt2_channel2, "qg (channel 2)", "f")
leg.AddEntry(hist_pt2_channel3, "gg (channel 3)", "f")
leg.Draw()

canvas.Modified()
canvas.Update()
canvas.WaitPrimitive()

hs_psi12.Draw("HIST")

leg = ROOT.TLegend(0.65, 0.70, 0.88, 0.88)  # (x1,y1,x2,y2) in NDC
leg.SetBorderSize(0)
leg.SetFillStyle(0)
leg.AddEntry(hist_psi12_channel1, "qq (channel 1)", "f")
leg.AddEntry(hist_psi12_channel2, "qg (channel 2)", "f")
leg.AddEntry(hist_psi12_channel3, "gg (channel 3)", "f")
leg.Draw()

canvas.Modified()
canvas.Update()
canvas.WaitPrimitive()

hist_z_channel1.Draw("COLZ")
canvas.Modified()
canvas.Update()
canvas.WaitPrimitive()

hist_z_channel2.Draw("COLZ")
canvas.Modified()
canvas.Update()
canvas.WaitPrimitive()

hist_z_channel3.Draw("COLZ")
canvas.Modified()
canvas.Update()
canvas.WaitPrimitive()

hs_pt1.Draw("HIST")

leg = ROOT.TLegend(0.65, 0.70, 0.88, 0.88)  # (x1,y1,x2,y2) in NDC
leg.SetBorderSize(0)
leg.SetFillStyle(0)
leg.AddEntry(hist_pt1_channel1, "qq (channel 1)", "f")
leg.AddEntry(hist_pt1_channel2, "qg (channel 2)", "f")
leg.AddEntry(hist_pt1_channel3, "gg (channel 3)", "f")
leg.Draw()

canvas.Modified()
canvas.Update()
canvas.WaitPrimitive()

hs_tau_time.Draw("HIST")

leg = ROOT.TLegend(0.65, 0.70, 0.88, 0.88)  # (x1,y1,x2,y2) in NDC
leg.SetBorderSize(0)
leg.SetFillStyle(0)
leg.AddEntry(hist_tau_time_channel1, "qq (channel 1)", "f")
leg.AddEntry(hist_tau_time_channel2, "qg (channel 2)", "f")
leg.AddEntry(hist_tau_time_channel3, "gg (channel 3)", "f")
leg.Draw()

canvas.Modified()
canvas.Update()
canvas.WaitPrimitive()