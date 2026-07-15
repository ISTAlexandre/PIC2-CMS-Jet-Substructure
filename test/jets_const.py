# -*- coding: utf-8 -*-
#mpiexec -n 2 python jets_const.py   
from __future__ import print_function
import ROOT
from DataFormats.FWLite import Events, Handle
import os

import FWCore.ParameterSet.Config as cms
import FWCore.PythonUtilities.LumiList as LumiList
from mpi4py import MPI
import json

try:
    unicode  # Python 2
except NameError:
    unicode = str

def to_bytes(s):
    return s.encode('utf-8') if isinstance(s, unicode) else s

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

local = True  # set False to fetch from opendata.cern.ch

min_pt = 600
max_eta = 1.7
const_min_pt = 1.0
phi_cut = 2.0

goodJSON = 'Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt'
myLumis = LumiList.LumiList(filename = goodJSON)

with open(goodJSON) as f:
    good_runs_lumis = json.load(f)

good_runs_lumis = {int(k): v for k,v in good_runs_lumis.items()}

def is_good_lumi(run, lumi):
    run = int(run)
    lumi = int(lumi)
    if run not in good_runs_lumis:
        return False
    for lo, hi in good_runs_lumis[run]:
        if lo <= lumi <= hi:
            return True
    return False

def delta_phi(phi1, phi2):
    dphi = phi1 - phi2
    while dphi > ROOT.TMath.Pi():
        dphi -= 2*ROOT.TMath.Pi()
    while dphi <= -ROOT.TMath.Pi():
        dphi += 2*ROOT.TMath.Pi()
    return abs(dphi)

out_folder = "out/"
txt_folder = "txt/"

#clean create output folder if necessary without showing errors
if rank ==0 and not os.path.isdir(out_folder):
    os.makedirs(out_folder)

my_files = []

if not local:
    json_files = [f for f in os.listdir(txt_folder) if f.endswith(".json")]
    for jf in json_files:
        with open(os.path.join(txt_folder, jf)) as f:
            data = json.load(f)

            # handle both shapes: {"files":[{"uri":...,"filename":...}, ...]}
            # or {"entries":[{"uri":...,"filename":...}, ...]}
            entries = data.get('files') or data.get('entries') or []
            for e in entries:
                uri = e.get('uri') or e.get('url')  # some dumps use "url"
                fname = e.get('filename') or os.path.basename(uri or '')
                if uri:
                    # ensure byte strings for PyROOT/FWLite (Python 2)
                    #my_files.append((to_bytes(uri), to_bytes(fname)))
                    my_files.append((to_bytes(uri), fname))   # keep fname as str

    my_files = my_files[:4]  #limit for testing

else:
    local_folder = "in/"
    local_files = [f for f in os.listdir(local_folder) if f.endswith(".root")]
    for lf in local_files:
        my_files.append((os.path.join(local_folder, lf), lf))

# MPI partition
my_files = [f for i, f in enumerate(my_files) if i % size == rank]
print("Total number of files: ", len(my_files), " for rank ", rank)

#wait for other ranks
comm.Barrier()
print("Rank ", rank, " starting processing")

for fileName in my_files:
    events = Events(fileName[0])
    print("Rank ", rank, " processing file: ", fileName[1], " with ", events.size(), " events")

    handleJets = Handle("std::vector<pat::Jet>")
    labelJets = ("slimmedJetsAK8")
    #labelJets = ("slimmedJets")  # change to slimmedJets for ak4PFJets

    handleTrig = Handle("edm::TriggerResults")
    labelTrig = ("TriggerResults", "", "HLT")

    outfile = ROOT.TFile(os.path.join(out_folder, "out_" + fileName[1]), "RECREATE")
    tree = ROOT.TTree("jetTree", "Tree with jet and constituent information")

    #storage variables
    #jets
    pt_jet = ROOT.std.vector('float')()
    eta_jet = ROOT.std.vector('float')()
    phi_jet = ROOT.std.vector('float')()
    mass_jet = ROOT.std.vector('float')()
    jetAK = ROOT.std.vector('int')()

    #constituents
    pt_const = ROOT.std.vector('std::vector<float>')()
    eta_const = ROOT.std.vector('std::vector<float>')()
    phi_const = ROOT.std.vector('std::vector<float>')()
    mass_const = ROOT.std.vector('std::vector<float>')()

    #branches
    tree.Branch("jet_pt",pt_jet)
    tree.Branch("jet_eta",eta_jet)
    tree.Branch("jet_phi",phi_jet)
    tree.Branch("jet_mass",mass_jet)
    tree.Branch("jetAK",jetAK)

    tree.Branch("const_pt",pt_const)
    tree.Branch("const_eta",eta_const)
    tree.Branch("const_phi",phi_const)
    tree.Branch("const_mass",mass_const)

    for i, event in enumerate(events):

        aux = event.eventAuxiliary()
        run  = aux.run()
        lumi = aux.luminosityBlock()

        if not is_good_lumi(run, lumi):
            continue
        
        event.getByLabel(labelTrig, handleTrig)
        triggerBits = handleTrig.product()
        names = event.object().triggerNames(triggerBits)

        pass_trigger = False
        for iTrig in range(triggerBits.size()):
            name = names.triggerName(iTrig)
            #if (name.startswith("HLT_PFJet450") or name.startswith("HLT_PFJet500")) and triggerBits.accept(iTrig):
            if name.startswith("HLT_AK8PFJet500") and triggerBits.accept(iTrig):
                pass_trigger = True
                break
        if not pass_trigger:
            continue

        event.getByLabel(labelJets, handleJets)
        jets = handleJets.product()

        if len(jets) == 0:
            continue        

        #clear vectors
        pt_jet.clear()
        eta_jet.clear()
        phi_jet.clear()
        mass_jet.clear()
        jetAK.clear()
        
        pt_const.clear()
        eta_const.clear()
        phi_const.clear()
        mass_const.clear()


        for jet in jets:
            if jet.pt() < min_pt or abs(jet.eta()) > max_eta:
                continue

            pt_jet.push_back(jet.pt())
            eta_jet.push_back(jet.eta())
            phi_jet.push_back(jet.phi())
            mass_jet.push_back(jet.mass())
            jetAK.push_back(8)  # change to 4 if using ak4PFJets

            #constituents
            sub_pt = ROOT.std.vector('float')()
            sub_eta = ROOT.std.vector('float')()
            sub_phi = ROOT.std.vector('float')()
            sub_mass = ROOT.std.vector('float')()

            for const in jet.getJetConstituents():
                if const.pt() < const_min_pt:
                    continue
                sub_pt.push_back(const.pt())
                sub_eta.push_back(const.eta())
                sub_phi.push_back(const.phi())
                sub_mass.push_back(const.mass())
            
            pt_const.push_back(sub_pt)
            eta_const.push_back(sub_eta)
            phi_const.push_back(sub_phi)
            mass_const.push_back(sub_mass)
        
        #At least two jets seperated by 2 radians
        delta_phi_ok = False
        for j1 in range(len(pt_jet)):
            for j2 in range(j1+1, len(pt_jet)):
                if delta_phi(phi_jet[j1], phi_jet[j2]) >= phi_cut:
                    delta_phi_ok = True
                    break
            if delta_phi_ok:
                break
        if not delta_phi_ok:
            continue

        tree.Fill()

    outfile.cd()
    tree.Write()
    outfile.Close()

    print("Rank ", rank, " finished file: ", fileName[1])

print("Rank ", rank, " finished processing")

#Merge output files into on file
 
# Wait for all ranks to finish writing their ROOT files
comm.Barrier()

# Merge output files into one file (rank 0 only)
if rank == 0:
    merged_path = os.path.join(out_folder, "merged.root")

    output_files = sorted(
        os.path.join(out_folder, f)
        for f in os.listdir(out_folder)
        if f.startswith("out_") and f.endswith(".root")
    )

    if len(output_files) == 0:
        raise RuntimeError("No output ROOT files found to merge in: {}".format(out_folder))

    # Prefer calling hadd with a list to avoid shell quoting issues
    import subprocess

    cmd = ["hadd", "-f", merged_path] + output_files
    print("Merging {} files -> {}".format(len(output_files), merged_path))
    subprocess.check_call(cmd)
