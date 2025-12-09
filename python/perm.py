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
    return sign * ang


file_path = "root/out_94C50CE8-43B0-AF4D-A8AE-BE0C7EC09B80.root"

file = ROOT.TFile.Open(file_path)
tree = file.Get("jetTree")

n_entries = tree.GetEntries()
if rank == 0:
    print(f"Number of entries in the tree: {n_entries}")
    print(f"Starting processing...")

comm.Barrier()

var_psi = []
weights = []
start_time = time.time()
entries_processed = 0

for i in range(rank, n_entries, size):
    if entries_processed % 100 == 0 and entries_processed > 0:
        break
        print(f"Processing entry {i}/{n_entries}, Time elapsed: {round((time.time()-start_time)/60, 2)} minutes")
    tree.GetEntry(i)
    for jet_i in range(len(tree.jet_pt)):

        jet_pt = tree.jet_pt[jet_i]
        if jet_pt < 200:
            continue

        if len(tree.const_mass[jet_i]) < 3:
            continue
        idx_triples = list(permutations(range(len(tree.const_mass[jet_i])), 3))
        
        for perm in idx_triples:
            pt1 = tree.const_pt[jet_i][perm[0]]
            pt2 = tree.const_pt[jet_i][perm[1]]
            pt3 = tree.const_pt[jet_i][perm[2]]

            phi1 = tree.const_phi[jet_i][perm[0]]
            phi2 = tree.const_phi[jet_i][perm[1]]
            phi3 = tree.const_phi[jet_i][perm[2]]

            eta1 = tree.const_eta[jet_i][perm[0]]
            eta2 = tree.const_eta[jet_i][perm[1]]
            eta3 = tree.const_eta[jet_i][perm[2]]

            mass1 = tree.const_mass[jet_i][perm[0]]
            mass2 = tree.const_mass[jet_i][perm[1]]
            mass3 = tree.const_mass[jet_i][perm[2]]

            p1 = vec_from_pt_eta_phi(pt1, eta1, phi1)
            p2 = vec_from_pt_eta_phi(pt2, eta2, phi2)
            p3 = vec_from_pt_eta_phi(pt3, eta3, phi3)

            dpsi = plane_angle_signed(p1, p2, p3)
            if dpsi == float('inf'):
                print("Warning: Zero vector encountered in plane_angle_signed calculation.")
                continue
            w = 8*pt1*pt2*pt3 / (jet_pt**3)
            var_psi.append(dpsi)
            weights.append(w)

    entries_processed += 1


hist = ROOT.TH1F("hist_psi", "Distribution of #psi;#psi;Entries", 50, -np.pi, np.pi)
hist.Sumw2()

for i in range(len(var_psi)):
    hist.Fill(var_psi[i], weights[i])

hist.Scale(1./hist.Integral())
canvas = ROOT.TCanvas("canvas", "Canvas for #psi", 800, 600)
hist.Draw("HIST")
canvas.SaveAs(f"imgs/psi_distribution_rank_{rank}.png")
print(f"Rank {rank} finished processing. Total time: {round((time.time()-start_time)/60, 2)} minutes")
comm.Barrier()