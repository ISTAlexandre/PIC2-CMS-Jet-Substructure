#mpiexec -n 2 python jets_cont.py   
from __future__ import print_function
import ROOT
from DataFormats.FWLite import Events, Handle
import pandas as pd
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

goodJSON = 'Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt'
myLumis = LumiList.LumiList(filename = goodJSON).getCMSSWString().split(',')

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
                    my_files.append((to_bytes(uri), to_bytes(fname)))

    my_files = my_files[:6]  #limit for testing

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

    handleJets = Handle("std::vector<pat::Jet>")
    labelJets = ("slimmedJets")
    labelJetsAK8 = ("slimmedJetsAK8")

    #Output root file
    outfile = ROOT.TFile(os.path.join(out_folder, "out_" + fileName[1]), "RECREATE")
    tree = ROOT.TTree("jetTree", "Tree with jet and constituent information")

    #storage variables
    #jets
    pt_jet = ROOT.std.vector('float')()
    eta_jet = ROOT.std.vector('float')()
    phi_jet = ROOT.std.vector('float')()
    mass_jet = ROOT.std.vector('float')()
    nJets = ROOT.std.vector('int')()
    jetAK = ROOT.std.vector('int')()
    jet_btag = ROOT.std.vector('float')()

    #constituents
    pt_const = ROOT.std.vector('std::vector<float>')()
    eta_const = ROOT.std.vector('std::vector<float>')()
    phi_const = ROOT.std.vector('std::vector<float>')()
    mass_const = ROOT.std.vector('std::vector<float>')()
    const_pdgId = ROOT.std.vector('std::vector<int>')()

    #branches
    tree.Branch("nJets",nJets)
    tree.Branch("jet_pt",pt_jet)
    tree.Branch("jet_eta",eta_jet)
    tree.Branch("jet_phi",phi_jet)
    tree.Branch("jet_mass",mass_jet)
    tree.Branch("jetAK",jetAK)
    tree.Branch("jet_btag",jet_btag)

    tree.Branch("const_pt",pt_const)
    tree.Branch("const_eta",eta_const)
    tree.Branch("const_phi",phi_const)
    tree.Branch("const_mass",mass_const)
    tree.Branch("const_pdgId",const_pdgId)

    maxEvents = -1
    print("Number of events: ", events.size(), " for rank ", rank)
    for i, event in enumerate(events):
        if maxEvents > 0 and i >= maxEvents:
            break
        if i %1000 ==0:
            print("Rank ", rank, " processing event ", i, " / ", events.size())

        event.getByLabel(labelJets, handleJets)
        jets = handleJets.product()
        event.getByLabel(labelJetsAK8, handleJets)
        jetsAK8 = handleJets.product()

        #clear all vectors
        pt_jet.clear()
        eta_jet.clear()
        phi_jet.clear()
        mass_jet.clear()
        nJets.clear()
        jetAK.clear()
        jet_btag.clear()

        pt_const.clear()
        eta_const.clear()
        phi_const.clear()
        mass_const.clear()
        const_pdgId.clear()

        nJets.push_back(len(jets)+len(jetsAK8))

        for jet in jets:

            pt_jet.push_back(jet.pt())
            eta_jet.push_back(jet.eta())
            phi_jet.push_back(jet.phi())
            mass_jet.push_back(jet.mass())
            jetAK.push_back(4)
            bscore = jet.bDiscriminator("pfCombinedInclusiveSecondaryVertexV2BJetTags")
            jet_btag.push_back(bscore)

            sub_pt = ROOT.std.vector('float')()
            sub_eta = ROOT.std.vector('float')()
            sub_phi = ROOT.std.vector('float')()
            sub_mass = ROOT.std.vector('float')()
            sub_pdgId = ROOT.std.vector('int')()

            for cand in jet.getJetConstituents():
                sub_pt.push_back(cand.pt())
                sub_eta.push_back(cand.eta())
                sub_phi.push_back(cand.phi())
                sub_mass.push_back(cand.mass())
                sub_pdgId.push_back(cand.pdgId())

            pt_const.push_back(sub_pt)
            eta_const.push_back(sub_eta)
            phi_const.push_back(sub_phi)
            mass_const.push_back(sub_mass)
            const_pdgId.push_back(sub_pdgId)
        
        '''
        for jet in jetsAK8:
            
            pt_jet.push_back(jet.pt())
            eta_jet.push_back(jet.eta())
            phi_jet.push_back(jet.phi())
            mass_jet.push_back(jet.mass())
            jetAK.push_back(8)

            sub_pt = ROOT.std.vector('float')()
            sub_eta = ROOT.std.vector('float')()
            sub_phi = ROOT.std.vector('float')()
            sub_mass = ROOT.std.vector('float')()
            sub_pdgId = ROOT.std.vector('int')()

            for cand in jet.getJetConstituents():
                sub_pt.push_back(cand.pt())
                sub_eta.push_back(cand.eta())
                sub_phi.push_back(cand.phi())
                sub_mass.push_back(cand.mass())
                sub_pdgId.push_back(cand.pdgId())

            pt_const.push_back(sub_pt)
            eta_const.push_back(sub_eta)
            phi_const.push_back(sub_phi)
            mass_const.push_back(sub_mass)
            const_pdgId.push_back(sub_pdgId)
        '''
        
        tree.Fill()

    outfile.cd()
    tree.Write()
    outfile.Close()

#comm.Barrier()
print("DONE", " for rank ", rank)

