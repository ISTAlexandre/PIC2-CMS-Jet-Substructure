# -*- coding: utf-8 -*-
from __future__ import print_function
import ROOT
from DataFormats.FWLite import Events, Handle
import os

import FWCore.ParameterSet.Config as cms
import FWCore.PythonUtilities.LumiList as LumiList
import json

try:
    unicode  # Python 2
except NameError:
    unicode = str

try:
    unicode  # Python 2
except NameError:
    unicode = str

def to_bytes(s):
    return s.encode('utf-8') if isinstance(s, unicode) else s

rank = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
size = int(os.environ.get('SLURM_ARRAY_TASK_COUNT', 1))

local = False  # set False to fetch from opendata.cern.ch

min_pt = 700
max_eta = 1.7
const_min_pt = 0.0
phi_cut = 2.0

goodJSON = 'Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt'
myLumis = LumiList.LumiList(filename = goodJSON)

out_folder = "out/"
txt_folder = "txt/"

with open(goodJSON) as f:
    good_runs_lumis = json.load(f)

def is_good_lumi(run, lumi):
    run = str(int(run))  # JSON keys are strings, ensure run is string
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

def delta_R(eta1, phi1, eta2, phi2):
    deta = eta1 - eta2
    dphi = delta_phi(phi1, phi2)
    return (deta**2 + dphi**2)**0.5

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

    #my_files = my_files[:4]  #limit for testing

else:
    local_folder = "in/"
    local_files = [f for f in os.listdir(local_folder) if f.endswith(".root")]
    for lf in local_files:
        my_files.append((os.path.join(local_folder, lf), lf))

# MPI partition
my_files = [f for i, f in enumerate(my_files) if i % size == rank]
print("Total number of files: ", len(my_files), " for rank ", rank)
print("Rank ", rank, " starting processing")

for fileName in my_files:
    events = Events(fileName[0])
    print("Rank ", rank, " processing file: ", fileName[1], " with ", events.size(), " events")

    #Skip empty files
    if events.size() == 0:
        print("Rank ", rank, " skipping empty file: ", fileName[1])
        continue

    handleJets = Handle("std::vector<pat::Jet>")
    labelJets = ("slimmedJetsAK8")

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
    charge_const = ROOT.std.vector('std::vector<int>')()

    #branches for jets
    tree.Branch("jet_pt",pt_jet)
    tree.Branch("jet_eta",eta_jet)
    tree.Branch("jet_phi",phi_jet)
    tree.Branch("jet_mass",mass_jet)
    tree.Branch("jetAK",jetAK)

    tree.Branch("const_pt",pt_const)
    tree.Branch("const_eta",eta_const)
    tree.Branch("const_phi",phi_const)
    tree.Branch("const_mass",mass_const)
    tree.Branch("const_charge",charge_const)

    for i, event in enumerate(events):
        if (i % 1000 == 0):
            print("Rank ", rank, " processed ", i, " / ", events.size(), " events of file: ", fileName[1])
        
        # check if event is in good lumi section
        aux = event.eventAuxiliary()
        run = aux.run()
        lumi = aux.luminosityBlock()
        if not is_good_lumi(run, lumi):
            continue
        
        print("passed lumi check")
        event.getByLabel(labelTrig, handleTrig)
        triggerBits = handleTrig.product()
        names = event.object().triggerNames(triggerBits)

        pass_trigger = False
        for iTrig in range(triggerBits.size()):
            name = names.triggerName(iTrig)
            #if (name.startswith("HLT_PFJet450") or name.startswith("HLT_PFJet500")) and triggerBits.accept(iTrig):
            if name.startswith("HLT_AK8PFJet") and triggerBits.accept(iTrig):
                pass_trigger = True
                break
        if not pass_trigger:
            continue
        print("passed trigger check")

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
        charge_const.clear()

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
            sub_charge = ROOT.std.vector('int')()

            for const in jet.getJetConstituents():
                if const.pt() < const_min_pt:
                    continue
                sub_pt.push_back(const.pt())
                sub_eta.push_back(const.eta())
                sub_phi.push_back(const.phi())
                sub_mass.push_back(const.mass())
                sub_charge.push_back(const.charge())
                        
            pt_const.push_back(sub_pt)
            eta_const.push_back(sub_eta)
            phi_const.push_back(sub_phi)
            mass_const.push_back(sub_mass)
            charge_const.push_back(sub_charge)
        
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