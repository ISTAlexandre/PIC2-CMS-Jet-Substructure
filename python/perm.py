import ROOT
import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations
from mpi4py import MPI
import time as time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def phi_of_sum(pt1,pt2,phi1,phi2):
    x = pt1 * np.cos(phi1) + pt2 * np.cos(phi2)
    y = pt1 * np.sin(phi1) + pt2 * np.sin(phi2)
    return np.arctan2(y, x)

def vec_from_pt_eta_phi(pt, eta, phi):
    px = pt*np.cos(phi)
    py = pt*np.sin(phi)
    pz = pt*np.sinh(eta)
    return np.array([px, py, pz], dtype=float)

def angle_between(p1,p2):
    n1 = np.linalg.norm(p1)
    n2 = np.linalg.norm(p2)
    if n1 == 0 or n2 == 0:
        return None
    return np.arccos(np.clip(np.dot(p1, p2) / (n1 * n2), -1.0, 1.0))

def plane_angle_signed(p1, p2, p3):
    n1 = np.cross(p1, p2)
    n2 = np.cross(p1 + p2, p3)
    n1_norm = np.linalg.norm(n1)
    n2_norm = np.linalg.norm(n2)
    if n1_norm == 0 or n2_norm == 0:
        return float('inf')
    cosang = np.clip(np.dot(n1, n2) / (n1_norm * n2_norm), -1.0, 1.0)
    ang = np.arccos(cosang)
    sign = np.sign(np.dot(np.cross(n1, n2), p1 + p2 + p3))
    return ang * sign


file_path = "root/out_94C50CE8-43B0-AF4D-A8AE-BE0C7EC09B80.root"

file = ROOT.TFile.Open(file_path)
tree = file.Get("jetTree")

n_entries = tree.GetEntries()

local_total = (n_entries - rank + size - 1) // size if n_entries > rank else 0
print(f"Rank {rank} will process {local_total} entries.")
comm.Barrier()

if rank == 0:
    print(f"Number of entries in the tree: {n_entries}")
    print(f"Starting processing...")

comm.Barrier()

start_time = time.time()
entries_processed = 0

hist = ROOT.TH1F("hist_psi", "Distribution of #psi;#psi;Entries", 50, -np.pi, np.pi)
hist.Sumw2()
histS = ROOT.TH1F("hist_thetaS", "Distribution of ThetaS", 50, 0, np.pi)
histL = ROOT.TH1F("hist_thetaL", "Distribution of ThetaL", 50, 0, np.pi)
histL12 = ROOT.TH1F("hist_thetaL12", "Distribution of ThetaL12", 50, 0, np.pi)

for i in range(rank, n_entries, size):
    #if entries_processed % 10 == 0 and entries_processed > 0:
    #    break
    tree.GetEntry(i)

    for jet_i in range(len(tree.jet_pt)):

        jet_pt = tree.jet_pt[jet_i]
        b_score = tree.jet_btag[jet_i]
        if jet_pt < 200: #or b_score < 0.8:
            continue

        if len(tree.const_mass[jet_i]) < 3:
            continue
        
        N = len(tree.const_pt[jet_i]) 
        for perm in permutations(range(N), 3):
            pt1 = tree.const_pt[jet_i][perm[0]]
            pt2 = tree.const_pt[jet_i][perm[1]]
            pt3 = tree.const_pt[jet_i][perm[2]]

            phi1 = tree.const_phi[jet_i][perm[0]]
            phi2 = tree.const_phi[jet_i][perm[1]]
            phi3 = tree.const_phi[jet_i][perm[2]]

            eta1 = tree.const_eta[jet_i][perm[0]]
            eta2 = tree.const_eta[jet_i][perm[1]]
            eta3 = tree.const_eta[jet_i][perm[2]]

            p1 = vec_from_pt_eta_phi(pt1, eta1, phi1)
            p2 = vec_from_pt_eta_phi(pt2, eta2, phi2)
            p3 = vec_from_pt_eta_phi(pt3, eta3, phi3)

            
            # Apply angular cuts
            thetaS = angle_between(p1, p2)
            thetaL = angle_between(p2, p3)
            thetaL12 = angle_between(p1 + p2, p3)

            '''
            if thetaS is None or thetaL is None:
                continue
            if not (0.01 < thetaS < 0.1):
                continue
            if not (np.sqrt(0.1) < thetaL < 1):
                continue
            '''

            dpsi = plane_angle_signed(p1, p2, p3)
            if dpsi == float('inf'):
                print("Warning: Zero vector encountered in plane_angle_signed calculation.")
                continue
            w = pt1*pt2*pt3 / (jet_pt**3) #*8
            hist.Fill(dpsi, w)
            histS.Fill(thetaS)
            histL.Fill(thetaL)
            histL12.Fill(thetaL12)

    entries_processed += 1

canvas = ROOT.TCanvas("canvas", "Canvas", 800, 600)
#hist.Scale(1.0 / hist.Integral())
hist.Draw("HIST")
canvas.SaveAs(f"imgs/psi_distribution_rank_{rank}.png")

comm.Barrier()

# Send histograms to root process
if rank != 0:
    comm.send(hist, dest=0, tag=77)
    comm.send(histS, dest=0, tag=78)
    comm.send(histL, dest=0, tag=79)
    comm.send(histL12, dest=0, tag=80)

if rank == 0:
    final_hist = hist.Clone()
    final_histS = histS.Clone()
    final_histL = histL.Clone()
    final_histL12 = histL12.Clone()

    for other_rank in range(1, size):
        temp_hist = comm.recv(source=other_rank, tag=77)
        temp_histS = comm.recv(source=other_rank, tag=78)
        temp_histL = comm.recv(source=other_rank, tag=79)
        temp_histL12 = comm.recv(source=other_rank, tag=80)
        final_hist.Add(temp_hist)
        final_histS.Add(temp_histS)
        final_histL.Add(temp_histL)
        final_histL12.Add(temp_histL12)

    # Normalize
    final_hist.Scale(1.0 / final_hist.Integral())
    final_hist.Draw("HIST")
    canvas.SaveAs("imgs/psi_distribution.png")

    final_histS.Draw("HIST")
    canvas.SaveAs("imgs/thetaS_distribution.png")

    final_histL.Draw("HIST")
    canvas.SaveAs("imgs/thetaL_distribution.png")

    final_histL12.Draw("HIST")
    canvas.SaveAs("imgs/thetaL12_distribution.png")

    print(f"Processing complete. Total time: {round((time.time()-start_time)/60, 2)} minutes")