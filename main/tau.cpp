/*
g++ -std=c++17 -O2 \
main/tau.cpp -o build/tau \
$(fastjet-config --cxxflags) $(root-config --cflags) \
$(fastjet-config --libs) -lfastjetplugins -lfastjettools -lfastjetcontribfragile \
$(root-config --libs)
*/

//FastJet includes
#include "fastjet/PseudoJet.hh"
#include "fastjet/EECambridgePlugin.hh"
#include "fastjet/contrib/LundGenerator.hh"
#include "fastjet/contrib/SoftDrop.hh"

//ROOT includes
#include "TFile.h"
#include "TTree.h"
#include "TKey.h"
#include <TTreeReader.h>
#include <TTreeReaderValue.h>
#include <TH1D.h>
#include <TH1F.h>
#include <TCanvas.h>

//C++ includes
#include <iostream>
#include <fstream>
#include <sstream>
#include <limits>
#include <memory>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>

//Namespaces
using namespace std;
using namespace fastjet;
using namespace fastjet::contrib;

double pseudo_dot_product(const PseudoJet& a, const PseudoJet& b){
    return a.px()*b.px() + a.py()*b.py() + a.pz()*b.pz();
}

double norm(const PseudoJet& a){
    return sqrt(pseudo_dot_product(a,a));
}

double cos_angle_between(const PseudoJet& a, const PseudoJet& b){
    double dot = pseudo_dot_product(a,b);
    double n_a = norm(a);
    double n_b = norm(b);
    if (n_a == 0 || n_b == 0){
        cout << "Warning: zero-norm vector in angle calculation, returning Nan\n";
        return numeric_limits<double>::quiet_NaN();;
    }
    double cos_theta = dot / (n_a * n_b);
    // Clamp to [-1,1] to avoid numerical issues with acos
    cos_theta = max(-1.0, min(1.0, cos_theta));
    return cos_theta;
}

double formation_time(const LundDeclustering& declust){
    const PseudoJet& p_hard = declust.harder();
    const PseudoJet& p_soft = declust.softer();

    const double energy = p_hard.E() + p_soft.E();
    if (energy <= 0.0){
        cout << "Warning: non-positive energy in formation time calculation, returning NaN\n";
        return numeric_limits<double>::quiet_NaN();
    }

    const double z1 = p_hard.E() / energy;
    const double z2 = p_soft.E() / energy;

    const double cos12 = cos_angle_between(p_hard, p_soft);
    if (!isfinite(cos12)){
        cout << "Warning: NaN in cos(angle) calculation, returning NaN for formation time\n";
        return numeric_limits<double>::quiet_NaN();
    }

    const double denom = 2.0 * energy * z1 * z2 * (1.0 - cos12);
    if (denom <= 0.0){
        cout << "Warning: non-positive denominator in formation time calculation, returning NaN\n";
        return numeric_limits<double>::quiet_NaN();
    }

    return 1.0 / denom;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        cerr << "Usage: ./lund_plane input.root [rank]\n";
        return 1;
    }
    const char* path = argv[1];
    int rank = (argc > 2) ? atoi(argv[2]) : -1;
    cout << "Processing file " << path << " (passed rank=" << rank << ")\n";

    // Open the ROOT file and get the TTree
    TFile* file = TFile::Open(path,"UPDATE");
    if (!file || file->IsZombie()) {
        cerr << "Error: could not open file " << path << endl;
        return 1;
    }
    TTree* tree = dynamic_cast<TTree*>(file->Get("jetTree"));
    if (!tree) {
        cerr << "Error: could not find TTree 'jetTree' in file " << path << endl;
        file->Close();
        return 1;
    }

    TTreeReader reader(tree);
    TTreeReaderValue<vector<float>> jet_pt(reader, "jet_pt");
    TTreeReaderValue<vector<vector<float>>> const_pt(reader, "const_pt");
    TTreeReaderValue<vector<vector<float>>> const_eta(reader, "const_eta");
    TTreeReaderValue<vector<vector<float>>> const_phi(reader, "const_phi");
    TTreeReaderValue<vector<vector<float>>> const_mass(reader, "const_mass");

    Long64_t nevents = tree->GetEntries();
    cout << "Number of events: " << nevents << endl;

    const double sd_beta = 0;     // 0 => mMDT // -1 => "traditional" soft drop (agressive)
    const double sd_zcut = 0.1;     // typical 0.05–0.2
    const double R0   = 1.0;     // usually = jet R
    const double p = 0.5;     // genkt algorithm exponent

    int event_count = 0;

    // Create ROOT branches for output
    vector< vector<double>> tau_time_events;
    auto tau_time_branch = tree->Branch("tau_time", &tau_time_events);
    vector<double> tau_time_jet;

    vector< vector<double>> tau_deltaR_events;
    auto tau_deltaR_branch = tree->Branch("tau_deltaR", &tau_deltaR_events);
    vector<double> tau_deltaR_jet;

    while (reader.Next()){
        tau_time_events.clear();
        tau_deltaR_events.clear();

        for (size_t ijet=0; ijet < jet_pt->size(); ++ijet) {
            tau_time_jet.clear();
            tau_deltaR_jet.clear();

            vector<PseudoJet> constituents;
            for (size_t iconst=0; iconst < const_pt->at(ijet).size(); ++iconst) {
                double pt = const_pt->at(ijet)[iconst];
                double eta = const_eta->at(ijet)[iconst];
                double phi = const_phi->at(ijet)[iconst];
                double mass = const_mass->at(ijet)[iconst];

                double px = pt * cos(phi);
                double py = pt * sin(phi);
                double pz = pt * sinh(eta);
                double E  = sqrt(px*px + py*py + pz*pz + mass*mass);

                PseudoJet p(px, py, pz, E);
                constituents.push_back(p);
            }

            if (constituents.size() < 2 ) {
                tau_time_events.push_back({});
                tau_deltaR_events.push_back({});
                continue;
            }

            LundGenerator lund;
            SoftDrop sd(sd_beta, sd_zcut,R0);
            JetDefinition jet_def(genkt_algorithm, R0, p);
            ClusterSequence cs(constituents, jet_def);
            auto jets = sorted_by_pt(cs.inclusive_jets());
            
            if (jets.size() == 0 || jets.size() > 1) {
                cout << "Warning: event has " << jets.size() << " jets, skipping" << endl;
                tau_time_events.push_back({});
                tau_deltaR_events.push_back({});
                continue;
            }

            const PseudoJet groomed_jet = sd(jets[0]);
            if (!groomed_jet.has_structure()) {
                tau_time_events.push_back({});
                tau_deltaR_events.push_back({});
                continue;
            }

            vector<LundDeclustering> declusts = lund.result(groomed_jet);

            for (LundDeclustering declust : declusts){
                double tau_time = formation_time(declust);
                tau_time_jet.push_back(tau_time);
                double deltaR = declust.Delta();
                tau_deltaR_jet.push_back(deltaR);
            }

            tau_time_events.push_back(tau_time_jet);
            tau_deltaR_events.push_back(tau_deltaR_jet);
        }

        tau_time_branch->Fill();
        tau_deltaR_branch->Fill();

        if (event_count%1000 == 0) {
            cout << "Rank " << rank << ": processed " << event_count << " events of " << nevents << endl;
        }
        event_count++;
    }

    // Write the tree and close the file
    tree->Write("", TObject::kOverwrite);
    file->Write("", TObject::kOverwrite);
    file->Close();

    cout << "Rank " << rank << ": finished processing " << event_count << " events.\n";
}



