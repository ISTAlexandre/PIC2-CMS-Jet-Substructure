import ROOT
import numpy as np

file = ROOT.TFile.Open("root_ML/merged_ML-600to800.root")
tree = file.Get("jetTree")

n_events = tree.GetEntries()
print(f"Number of events in the tree: {n_events}")
ch1_list = [0,0,0,0,0]
ch2_list = [0,0,0,0,0]
cos2_list = [[[] for x in range(5)] for y in range(5)]
canvas = ROOT.TCanvas("canvas", "Canvas for Heatmap", 800, 600)
hist_heatmap = ROOT.TH2F("hist_heatmap", "Heatmap of channel1 and channel2", 5, 0, 5, 5, 0, 5)
# Take stat box out
hist_heatmap.SetStats(0)

for i in range(n_events):
    tree.GetEntry(i)
    
    for j in range(len(tree.jet_pt)):
        
        max_kt_index1 = int(tree.lund_max_kt_sd[j])
        max_kt_index2 = int(tree.lund_max_kt_secondary_sd[j])

        if max_kt_index1 < 0 or max_kt_index2 < 0:
            continue

        ch1 = tree.lund_primary_idx_sd[j]
        ch2 = tree.lund_secondary_idx_sd[j]
        hist_heatmap.Fill(ch1, ch2)

        if ch1 != -1:
            ch1_list[ch1] += 1
        
        if ch2 != -1:
            ch2_list[ch2] += 1
        
        if ch1 != -1 and ch2 != -1:
            cos2_list[ch1][ch2].append(np.cos(2*tree.lund_psi12_sd[j]))


ch1_total = sum(ch1_list)
ch2_total = sum(ch2_list)

for n in range(len(ch1_list)):
    ch1_percentage = (ch1_list[n] / ch1_total) * 100 if ch1_total > 0 else 0
    ch2_percentage = (ch2_list[n] / ch2_total) * 100 if ch2_total > 0 else 0
    print(f"Channel {n}: Primary Count = {ch1_list[n]}, Percentage = {ch1_percentage:.2f}%, Secondary Count = {ch2_list[n]}, Percentage = {ch2_percentage:.2f}%")

for n1 in range(len(cos2_list)):
    for n2 in range(len(cos2_list[n1])):
        if len(cos2_list[n1][n2]) > 0:
            avg_cos2 = np.mean(cos2_list[n1][n2])
            print(f"Channel {n1} to Channel {n2}: Average cos(theta) = {avg_cos2:.4f}, Count = {len(cos2_list[n1][n2])}")

hist_heatmap.GetXaxis().SetTitle("Channel 1")
hist_heatmap.GetYaxis().SetTitle("Channel 2")
hist_heatmap.Draw("COLZ")
canvas.Draw()
canvas.SaveAs("imgs/channel_heatmap.png")
canvas.WaitPrimitive()  # Wait for user input before closing the canvas
