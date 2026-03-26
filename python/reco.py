# -*- coding: utf-8 -*-
#mpiexec -n 4 python reco.py   
from __future__ import print_function
import ROOT
from DataFormats.FWLite import Events, Handle
#import pandas as pd
import os

import FWCore.ParameterSet.Config as cms
import FWCore.PythonUtilities.LumiList as LumiList
import json
from array import array
import math

try:
    unicode  # Python 2
except NameError:
    unicode = str

goodJSON = "Cert_181530-183126_HI7TeV_PromptReco_Collisions11_JSON.txt"
myLumis = LumiList.LumiList(filename = goodJSON).getCMSSWString().split(',')
out_folder = "out/"
txt_folder = "txt/"

#clean create output folder if necessary without showing errors
if not os.path.isdir(out_folder):
    os.makedirs(out_folder)

my_files = []
local_folder = "in/"
local_files = [f for f in os.listdir(local_folder) if f.endswith(".root")]
for lf in local_files:
    my_files.append((os.path.join(local_folder, lf), lf))
    #print((os.path.join(local_folder, lf), lf))

for fileName in my_files:
    events = Events(fileName[0])

    # RECO/AOD PFJets
    handlejets = Handle("std::vector<reco::PFJet>")
    labeljets = ("ak5PFJets") 

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

    #constituents
    pt_const = ROOT.std.vector('std::vector<float>')()
    eta_const = ROOT.std.vector('std::vector<float>')()
    phi_const = ROOT.std.vector('std::vector<float>')()
    mass_const = ROOT.std.vector('std::vector<float>')()

    #branches
    tree.Branch("nJets",nJets)
    tree.Branch("jet_pt",pt_jet)
    tree.Branch("jet_eta",eta_jet)
    tree.Branch("jet_phi",phi_jet)
    tree.Branch("jet_mass",mass_jet)
    tree.Branch("jetAK",jetAK)

    tree.Branch("const_pt",pt_const)
    tree.Branch("const_eta",eta_const)
    tree.Branch("const_phi",phi_const)
    tree.Branch("const_mass",mass_const)

    maxEvents = -1

    for i, event in enumerate(events):
        if maxEvents > 0 and i >= maxEvents:
            break
        if i %10 == 0:
            print("Processing event ", i, " / ", events.size())

        event.getByLabel(labeljets, handlejets)
        jets = handlejets.product()

        #clear vectors
        pt_jet.clear()
        eta_jet.clear()
        phi_jet.clear()
        mass_jet.clear()
        nJets.clear()
        jetAK.clear()
        
        pt_const.clear()
        eta_const.clear()
        phi_const.clear()
        mass_const.clear()

        nJets.push_back(len(jets))

        for jet in jets:
            pt_jet.push_back(jet.pt())
            eta_jet.push_back(jet.eta())
            phi_jet.push_back(jet.phi())
            mass_jet.push_back(jet.mass())
            jetAK.push_back(4)  # change to 4 if using ak4PFJets

            #constituents
            sub_pt = ROOT.std.vector('float')()
            sub_eta = ROOT.std.vector('float')()
            sub_phi = ROOT.std.vector('float')()
            sub_mass = ROOT.std.vector('float')()

            for const in jet.getPFConstituents():
                sub_pt.push_back(const.pt())
                sub_eta.push_back(const.eta())
                sub_phi.push_back(const.phi())
                sub_mass.push_back(const.mass())
            
            pt_const.push_back(sub_pt)
            eta_const.push_back(sub_eta)
            phi_const.push_back(sub_phi)
            mass_const.push_back(sub_mass)
        
        tree.Fill()

    outfile.cd()
    tree.Write()
    outfile.Close()

print("Finished processing all files.")

