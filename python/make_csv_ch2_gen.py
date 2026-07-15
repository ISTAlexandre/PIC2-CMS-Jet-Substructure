import ROOT
import os
import pandas as pd

rows = []

my_files = []
files_path = "root_ML"

for file in os.listdir(files_path):
    if file.endswith(".root"):
        my_files.append(os.path.join(files_path, file))

event_count = 0
for file in my_files:
    root_file = ROOT.TFile.Open(file)
    tree = root_file.Get("jetTree")

    n_entries = tree.GetEntries()
    print(f"Processing file: {file}, Number of entries: {n_entries}")
    
    for i in range(n_entries):
        if (i % 1000 == 0):
            print(f"Processed {i} / {n_entries} entries in file: {file}")
        tree.GetEntry(i)
        
        n_jets = tree.jet_pt.size()

        for jet_i in range(n_jets):

            channel2 = tree.lund_secondary_idx_sd[jet_i]
            channel1 = tree.lund_primary_idx_sd[jet_i]
            max_kt_index1 = int(tree.lund_max_kt_sd[jet_i])
            max_kt_index2 = int(tree.lund_max_kt_secondary_sd[jet_i])


            if channel2 == -1 or max_kt_index1 < 0 or max_kt_index2 < 0 or channel1 == -1:
                continue

            #variables to store in csv
            z1 = tree.lund_z_sd[jet_i][max_kt_index1]
            z2 = tree.lund_z_secondary_sd[jet_i][max_kt_index2]
            deltaR34 = tree.lund_delta_secondary_sd[jet_i][max_kt_index2]
            kt2 = tree.lund_kt_secondary_sd[jet_i][max_kt_index2]
            N_charged3 = tree.lund_p3_n_charged[jet_i]
            N_charged4 = tree.lund_p4_n_charged[jet_i]
            N_all3 = tree.lund_p3_n_all[jet_i]
            N_all4 = tree.lund_p4_n_all[jet_i]
            pt_weight3 = tree.lund_p3_sigma[jet_i]
            pt_weight4 = tree.lund_p4_sigma[jet_i]
            pt_dispersion3 = tree.lund_p3_ptD[jet_i] 
            pt_dispersion4 = tree.lund_p4_ptD[jet_i]
            dpsi12 = tree.lund_psi12_sd[jet_i]

            has_tau_time = len(tree.tau_time[jet_i]) > 0
            tau_time = None
            if has_tau_time:
                tau_time = tree.tau_time[jet_i][0]

            row = {
                "Event": event_count,
                "Jet": jet_i,
                "Jet_pt": tree.jet_pt[jet_i],
                "z1": z1,
                "z2": z2,
                "deltaR34": deltaR34,
                "kt2": kt2,
                "N_charged3": N_charged3,
                "N_charged4": N_charged4,
                "N_all3": N_all3,
                "N_all4": N_all4,
                "pt_weight3": pt_weight3,
                "pt_weight4": pt_weight4,
                "pt_dispersion3": pt_dispersion3,
                "pt_dispersion4": pt_dispersion4,
                "tau_time": tau_time,
                "channel2": channel2,
                "dpsi12": dpsi12
            }

            rows.append(row)
        
        event_count += 1

df = pd.DataFrame(rows)
df.to_csv("csv/merged_ML_ch2.csv", index=False)