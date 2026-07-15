# -*- coding: utf-8 -*-
# AOD equivalent of jets_gen.py
# Uses full genParticles collection (status 71-79 shower partons available)
# mpiexec -n 2 python jets_gen_aod.py

from __future__ import print_function
import ROOT
from DataFormats.FWLite import Events, Handle
import os
from mpi4py import MPI
import json

try:
    unicode
except NameError:
    unicode = str

def to_bytes(s):
    return s.encode('utf-8') if isinstance(s, unicode) else s

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

local = True  # set False to use json file list

min_pt      = 700.0
max_eta     = 1.7
const_min_pt = 1.0
phi_cut     = 2.0

# ── Helpers ──────────────────────────────────────────────────────────────────

def delta_phi(phi1, phi2):
    dphi = phi1 - phi2
    while dphi >  ROOT.TMath.Pi(): dphi -= 2*ROOT.TMath.Pi()
    while dphi <= -ROOT.TMath.Pi(): dphi += 2*ROOT.TMath.Pi()
    return abs(dphi)

def delta_R(eta1, phi1, eta2, phi2):
    deta = eta1 - eta2
    dphi = delta_phi(phi1, phi2)
    return (deta**2 + dphi**2)**0.5

# ── File list ─────────────────────────────────────────────────────────────────

out_folder = "out_ML_aod/"
txt_folder = "txt_ML/"

if rank == 0 and not os.path.isdir(out_folder):
    os.makedirs(out_folder)

my_files = []

if not local:
    json_files = [f for f in os.listdir(txt_folder) if f.endswith(".json")]
    for jf in json_files:
        with open(os.path.join(txt_folder, jf)) as f:
            data = json.load(f)
        entries = data.get('files') or data.get('entries') or []
        for e in entries:
            uri   = e.get('uri') or e.get('url')
            fname = e.get('filename') or os.path.basename(uri or '')
            if uri:
                my_files.append((to_bytes(uri), fname))
else:
    local_folder = "in_ML_aod/"
    for lf in sorted(os.listdir(local_folder)):
        if lf.endswith(".root"):
            my_files.append((os.path.join(local_folder, lf), lf))

my_files = [f for i, f in enumerate(my_files) if i % size == rank]
print("Rank", rank, "processing", len(my_files), "files")
comm.Barrier()

# ── Main loop ─────────────────────────────────────────────────────────────────

for fileName in my_files:
    events = Events(fileName[0])
    print("Rank", rank, "file:", fileName[1], "events:", events.size())

    # ── Handles ──────────────────────────────────────────────────────────────
    # AOD jet collection: ak8PFJetsCHS (equivalent to slimmedJetsAK8 in MiniAOD)
    handleJets = Handle("std::vector<reco::PFJet>")
    labelJets  = ("ak8PFJetsCHS", "", "RECO")

    # AOD PF candidates: full particleFlow collection (used to get jet constituents)
    handlePF   = Handle("std::vector<reco::PFCandidate>")
    labelPF    = ("particleFlow", "", "RECO")

    # AOD gen particles: FULL shower — status 71-79 partons ARE stored here
    # This is the key advantage over MiniAOD's prunedGenParticles
    handleGen  = Handle("std::vector<reco::GenParticle>")
    labelGen   = ("genParticles", "", "HLT")  # or "SIM" depending on campaign

    # Gen jets (AK8) — used to require a gen-jet match (quality cut)
    handleGenJets = Handle("std::vector<reco::GenJet>")
    labelGenJets  = ("ak8GenJetsNoNu", "", "HLT")  # NoNu = neutrinos excluded

    # ── Output tree ──────────────────────────────────────────────────────────
    outfile = ROOT.TFile(os.path.join(out_folder, "out_" + fileName[1]), "RECREATE")
    tree    = ROOT.TTree("jetTree", "Jet and constituent information (AOD)")

    # Jets
    pt_jet   = ROOT.std.vector('float')()
    eta_jet  = ROOT.std.vector('float')()
    phi_jet  = ROOT.std.vector('float')()
    mass_jet = ROOT.std.vector('float')()
    jetAK    = ROOT.std.vector('int')()

    # Jet constituents (charged PF candidates, pT > 1 GeV)
    pt_const   = ROOT.std.vector('std::vector<float>')()
    eta_const  = ROOT.std.vector('std::vector<float>')()
    phi_const  = ROOT.std.vector('std::vector<float>')()
    mass_const = ROOT.std.vector('std::vector<float>')()

    # Event-wide final-state gen partons (status 71-79 — full shower available in AOD)
    pt_gen    = ROOT.std.vector('float')()
    eta_gen   = ROOT.std.vector('float')()
    phi_gen   = ROOT.std.vector('float')()
    mass_gen  = ROOT.std.vector('float')()
    pdgId_gen = ROOT.std.vector('int')()

    tree.Branch("jet_pt",    pt_jet)
    tree.Branch("jet_eta",   eta_jet)
    tree.Branch("jet_phi",   phi_jet)
    tree.Branch("jet_mass",  mass_jet)
    tree.Branch("jetAK",     jetAK)
    tree.Branch("const_pt",  pt_const)
    tree.Branch("const_eta", eta_const)
    tree.Branch("const_phi", phi_const)
    tree.Branch("const_mass",mass_const)
    tree.Branch("gen_pt",    pt_gen)
    tree.Branch("gen_eta",   eta_gen)
    tree.Branch("gen_phi",   phi_gen)
    tree.Branch("gen_mass",  mass_gen)
    tree.Branch("gen_pdgId", pdgId_gen)

    for i, event in enumerate(events):

        if i % 1000 == 0:
            print("Rank", rank, ":", i, "/", events.size(), fileName[1])

        event.getByLabel(labelJets,    handleJets)
        event.getByLabel(labelPF,      handlePF)
        event.getByLabel(labelGen,     handleGen)
        event.getByLabel(labelGenJets, handleGenJets)

        jets      = handleJets.product()
        pf_cands  = handlePF.product()
        gen_parts = handleGen.product()
        gen_jets  = handleGenJets.product()

        if jets.size() == 0:
            continue

        # ── Clear ──────────────────────────────────────────────────────────
        pt_jet.clear();  eta_jet.clear();  phi_jet.clear();  mass_jet.clear()
        jetAK.clear()
        pt_const.clear(); eta_const.clear(); phi_const.clear(); mass_const.clear()
        pt_gen.clear();  eta_gen.clear();  phi_gen.clear()
        mass_gen.clear(); pdgId_gen.clear()

        # ── Jet loop ──────────────────────────────────────────────────────
        for jet in jets:
            if jet.pt() < min_pt or abs(jet.eta()) > max_eta:
                continue

            # Require a matched gen jet (same quality cut as MiniAOD version)
            # Match by dR < 0.4 to the closest ak8GenJetsNoNu jet
            best_gen_dR = 0.4
            matched_gen_jet = None
            for gj in gen_jets:
                dR = delta_R(jet.eta(), jet.phi(), gj.eta(), gj.phi())
                if dR < best_gen_dR:
                    best_gen_dR = dR
                    matched_gen_jet = gj
            if matched_gen_jet is None:
                continue  # no gen-jet match → skip (same as jet.genJet() in MiniAOD)

            pt_jet.push_back(jet.pt())
            eta_jet.push_back(jet.eta())
            phi_jet.push_back(jet.phi())
            mass_jet.push_back(jet.mass())
            jetAK.push_back(8)

            # ── Constituents ──────────────────────────────────────────────
            # In AOD, reco::PFJet stores constituent indices into particleFlow.
            # Use jet.getPFConstituents() to get them directly.
            sub_pt   = ROOT.std.vector('float')()
            sub_eta  = ROOT.std.vector('float')()
            sub_phi  = ROOT.std.vector('float')()
            sub_mass = ROOT.std.vector('float')()

            for cref in jet.getPFConstituents():
                const = cref.get()
                if const.pt() < const_min_pt:
                    continue
                # CMS uses charged particles only for substructure (like MiniAOD version)
                # charge() != 0 selects tracks; remove this line to use neutral+charged
                if const.charge() == 0:
                    continue
                sub_pt.push_back(const.pt())
                sub_eta.push_back(const.eta())
                sub_phi.push_back(const.phi())
                sub_mass.push_back(const.mass())

            pt_const.push_back(sub_pt)
            eta_const.push_back(sub_eta)
            phi_const.push_back(sub_phi)
            mass_const.push_back(sub_mass)

        # ── Gen parton loop (event-wide) ──────────────────────────────────
        # KEY ADVANTAGE of AOD: genParticles has the full Pythia8 shower.
        # Status 71-79 partons (final shower, pre-hadronization) ARE stored.
        # This directly fixes the MiniAOD prunedGenParticles coverage problem.
        for gp in gen_parts:
            absID = abs(gp.pdgId())

            # Quarks (1-6) and gluons (21) only
            if not ((1 <= absID <= 6) or absID == 21):
                continue

            # pT > 1 GeV (paper's cut)
            if gp.pt() < const_min_pt:
                continue

            # Status 71-79: final shower partons in Pythia8 (pre-hadronization)
            # These are reliably available in AOD genParticles unlike MiniAOD
            st = abs(gp.status())
            if not (71 <= st <= 79):
                continue

            # Additional safety: no further parton daughters
            # (handles edge cases where status codes overlap)
            hasPartonDaughter = False
            for d in xrange(gp.numberOfDaughters()):
                dauID = abs(gp.daughter(d).pdgId())
                if (1 <= dauID <= 6) or dauID == 21:
                    hasPartonDaughter = True
                    break
            if hasPartonDaughter:
                continue

            pt_gen.push_back(gp.pt())
            eta_gen.push_back(gp.eta())
            phi_gen.push_back(gp.phi())
            mass_gen.push_back(gp.mass())
            pdgId_gen.push_back(gp.pdgId())

        # ── Dijet delta-phi cut ───────────────────────────────────────────
        delta_phi_ok = False
        for j1 in xrange(pt_jet.size()):
            for j2 in xrange(j1+1, pt_jet.size()):
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
    print("Rank", rank, "finished:", fileName[1])

print("Rank", rank, "done")

# ── Merge ─────────────────────────────────────────────────────────────────────
comm.Barrier()
if rank == 0:
    import subprocess
    merged = os.path.join(out_folder, "merged.root")
    out_files = sorted(
        os.path.join(out_folder, f)
        for f in os.listdir(out_folder)
        if f.startswith("out_") and f.endswith(".root")
    )
    if not out_files:
        raise RuntimeError("No output files to merge in: " + out_folder)
    subprocess.check_call(["hadd", "-f", merged] + out_files)
    print("Merged →", merged)
