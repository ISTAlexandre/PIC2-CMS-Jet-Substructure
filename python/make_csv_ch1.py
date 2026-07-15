import ROOT
import os
import pandas as pd

rows = []

my_files = []
files_path = "root_ML"

for file in os.listdir(files_path):
    if file.endswith(".root"):
        my_files.append(os.path.join(files_path, file))

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
            deltaR12 = tree.lund_delta_sd[jet_i][max_kt_index1]
            kt1 = tree.lund_kt_sd[jet_i][max_kt_index1]
            kt2 = tree.lund_kt_secondary_sd[jet_i][max_kt_index2]
            N_charged1 = tree.lund_p1_n_charged[jet_i]
            N_charged2 = tree.lund_p2_n_charged[jet_i]
            N_all1 = tree.lund_p1_n_all[jet_i]
            N_all2 = tree.lund_p2_n_all[jet_i]
            pt_weight1 = tree.lund_p1_sigma[jet_i]
            pt_weight2 = tree.lund_p2_sigma[jet_i]
            pt_dispersion1 = tree.lund_p1_ptD[jet_i] 
            pt_dispersion2 = tree.lund_p2_ptD[jet_i]
            dpsi12 = tree.lund_psi12_sd[jet_i]

            has_tau_time = len(tree.tau_time[jet_i]) > 0
            tau_time = None
            if has_tau_time:
                tau_time = tree.tau_time[jet_i][0]

            row = {
                "Event": i,
                "Jet": jet_i,
                "z1": z1,
                "z2": z2,
                "deltaR12": deltaR12,
                "kt1": kt1,
                "kt2": kt2,
                "N_charged1": N_charged1,
                "N_charged2": N_charged2,
                "N_all1": N_all1,
                "N_all2": N_all2,
                "pt_weight1": pt_weight1,
                "pt_weight2": pt_weight2,
                "pt_dispersion1": pt_dispersion1,
                "pt_dispersion2": pt_dispersion2,
                "tau_time": tau_time,
                "channel1": channel1,
                "dpsi12": dpsi12
            }

            rows.append(row)

df = pd.DataFrame(rows)
df.to_csv("csv/merged_ML_ch1.csv", index=False)