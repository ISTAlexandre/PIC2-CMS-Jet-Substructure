/*
g++ -std=c++17 -O2 \
main/lund_softdrop.cpp -o build/lund_softdrop \
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

static inline double wrap_pm_pi(double x) {
    // Wrap x into (-pi, pi]
    const double two_pi = 2.0 * M_PI;
    x = std::fmod(x + M_PI, two_pi);
    if (x < 0) x += two_pi;
    x -= M_PI;
  
    // Make endpoint convention (-pi, pi]
    if (x <= -M_PI) x += two_pi;
    return x;
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

    int event_count = 0;

    // Create ROOT branches for output
    vector< vector<double> > lund_coords_events_x_sd; //Primary plane coordinates
    vector< vector<double> > lund_coords_events_y_sd;
    vector< vector<double> > lund_kt_events_sd;
    vector< vector<double> > lund_z_events_sd;
    vector< vector<double> > lund_psi_events_sd;
    vector< vector<double> > lund_mass_events_sd;

    vector< vector<double> > lund_coords_events_secondary_x_sd; //Secondary plane coordinates
    vector< vector<double> > lund_coords_events_secondary_y_sd;
    vector< vector<double> > lund_kt_events_secondary_sd;
    vector< vector<double> > lund_z_events_secondary_sd;
    vector< vector<double> > lund_psi_events_secondary_sd;
    vector< vector<double> > lund_mass_events_secondary_sd;

    vector< double> lund_psi12_events_sd; //Delta psi12 between primary and secondary planes

    //Setup ROOT branches
    auto lund_branch_x_sd = tree->Branch("lund_coords_x_sd", &lund_coords_events_x_sd);
    auto lund_branch_y_sd = tree->Branch("lund_coords_y_sd", &lund_coords_events_y_sd);
    auto lund_branch_kt_sd = tree->Branch("lund_kt_sd", &lund_kt_events_sd);
    auto lund_branch_z_sd = tree->Branch("lund_z_sd", &lund_z_events_sd);
    auto lund_branch_psi_sd = tree->Branch("lund_psi_sd", &lund_psi_events_sd);
    auto lund_branch_mass_sd = tree->Branch("lund_mass_sd", &lund_mass_events_sd);
    
    auto lund_branch_secondary_x_sd = tree->Branch("lund_coords_secondary_x_sd", &lund_coords_events_secondary_x_sd);
    auto lund_branch_secondary_y_sd = tree->Branch("lund_coords_secondary_y_sd", &lund_coords_events_secondary_y_sd);
    auto lund_branch_secondary_kt_sd = tree->Branch("lund_kt_secondary_sd", &lund_kt_events_secondary_sd);
    auto lund_branch_secondary_z_sd = tree->Branch("lund_z_secondary_sd", &lund_z_events_secondary_sd);
    auto lund_branch_secondary_psi_sd = tree->Branch("lund_psi_secondary_sd", &lund_psi_events_secondary_sd);
    auto lund_branch_secondary_mass_sd = tree->Branch("lund_mass_secondary_sd", &lund_mass_events_secondary_sd);

    auto lund_branch_psi12_sd = tree->Branch("lund_psi12_sd", &lund_psi12_events_sd);

    // Working vectors for jet-level declustering (reset for each jet)
    vector<double> lund_coords_jet_x;
    vector<double> lund_coords_jet_y;
    vector<double> lund_kt_jet;
    vector<double> lund_z_jet;
    vector<double> lund_psi_jet;
    vector<double> lund_mass_jet;

    while (reader.Next()){
        //Clear branches
        lund_coords_events_x_sd.clear();
        lund_coords_events_y_sd.clear();
        lund_kt_events_sd.clear();
        lund_z_events_sd.clear();
        lund_psi_events_sd.clear();
        lund_mass_events_sd.clear();

        lund_coords_events_secondary_x_sd.clear();
        lund_coords_events_secondary_y_sd.clear();
        lund_kt_events_secondary_sd.clear();
        lund_z_events_secondary_sd.clear();
        lund_psi_events_secondary_sd.clear();
        lund_mass_events_secondary_sd.clear();

        lund_psi12_events_sd.clear();

        for (size_t ijet=0; ijet < jet_pt->size(); ++ijet) {
            //cout << "Number of constituents in jet " << ijet << ": " << const_pt->at(ijet).size() << endl;
            vector<fastjet::PseudoJet> constituents;
            for (size_t iconst=0; iconst < const_pt->at(ijet).size(); ++iconst) {
                double pt = const_pt->at(ijet)[iconst];
                double eta = const_eta->at(ijet)[iconst];
                double phi = const_phi->at(ijet)[iconst];
                double mass = const_mass->at(ijet)[iconst];

                double px = pt * cos(phi);
                double py = pt * sin(phi);
                double pz = pt * sinh(eta);
                double E  = sqrt(px*px + py*py + pz*pz + mass*mass);

                fastjet::PseudoJet p(px, py, pz, E);
                constituents.push_back(p);
            }

            lund_coords_jet_x.clear();
            lund_coords_jet_y.clear();
            lund_kt_jet.clear();
            lund_z_jet.clear();
            lund_psi_jet.clear();
            lund_mass_jet.clear();

            if (constituents.size() < 2) {
                // Not enough constituents to decluster
                lund_coords_events_x_sd.push_back({});
                lund_coords_events_y_sd.push_back({});
                lund_kt_events_sd.push_back({});
                lund_z_events_sd.push_back({});
                lund_psi_events_sd.push_back({});
                lund_mass_events_sd.push_back({});

                lund_coords_events_secondary_x_sd.push_back({});
                lund_coords_events_secondary_y_sd.push_back({});
                lund_kt_events_secondary_sd.push_back({});
                lund_z_events_secondary_sd.push_back({});
                lund_psi_events_secondary_sd.push_back({});
                lund_mass_events_secondary_sd.push_back({});

                lund_psi12_events_sd.push_back(numeric_limits<double>::quiet_NaN());
                continue;
            }

            //Create lund generator and soft drop declusterer
            fastjet::contrib::LundGenerator lund;
            fastjet::JetDefinition jet_def(fastjet::cambridge_aachen_algorithm, R0);
            fastjet::contrib::SoftDrop sd(sd_beta, sd_zcut, R0);
            fastjet::ClusterSequence cs(constituents, jet_def);
            auto jets = fastjet::sorted_by_pt(cs.inclusive_jets());

            const fastjet::PseudoJet groomed_jet = sd(jets[0]);
            if (!groomed_jet.has_structure()) {
                lund_coords_events_x_sd.push_back({});
                lund_coords_events_y_sd.push_back({});
                lund_kt_events_sd.push_back({});
                lund_z_events_sd.push_back({});
                lund_psi_events_sd.push_back({});
                lund_mass_events_sd.push_back({});

                lund_coords_events_secondary_x_sd.push_back({});
                lund_coords_events_secondary_y_sd.push_back({});
                lund_kt_events_secondary_sd.push_back({});
                lund_z_events_secondary_sd.push_back({});
                lund_psi_events_secondary_sd.push_back({});
                lund_mass_events_secondary_sd.push_back({});

                lund_psi12_events_sd.push_back(numeric_limits<double>::quiet_NaN());
                continue;
            }

            vector<fastjet::contrib::LundDeclustering> declusters = lund(groomed_jet);

            if (declusters.size() == 0) {
                lund_coords_events_x_sd.push_back({});
                lund_coords_events_y_sd.push_back({});
                lund_kt_events_sd.push_back({});
                lund_z_events_sd.push_back({});
                lund_psi_events_sd.push_back({});
                lund_mass_events_sd.push_back({});

                lund_coords_events_secondary_x_sd.push_back({});
                lund_coords_events_secondary_y_sd.push_back({});
                lund_kt_events_secondary_sd.push_back({});
                lund_z_events_secondary_sd.push_back({});
                lund_psi_events_secondary_sd.push_back({});
                lund_mass_events_secondary_sd.push_back({});

                lund_psi12_events_sd.push_back(numeric_limits<double>::quiet_NaN());
                continue;
            }

            double max_kt = -1.0;
            int max_kt_index = -1;
            int i_declust = 0;
            for (fastjet::contrib::LundDeclustering declust : declusters) {
                pair<double,double> coords = declust.lund_coordinates();
                double kt = declust.kt();
                double z = declust.z();
                double psi = declust.psi();
                double mass = declust.m();

                if (kt > max_kt) {
                    max_kt = kt;
                    max_kt_index = i_declust;
                }
                i_declust++;

                lund_coords_jet_x.push_back(coords.first);
                lund_coords_jet_y.push_back(coords.second);
                lund_kt_jet.push_back(kt);
                lund_z_jet.push_back(z);
                lund_psi_jet.push_back(psi);
                lund_mass_jet.push_back(mass);
            }

            lund_coords_events_x_sd.push_back(lund_coords_jet_x);
            lund_coords_events_y_sd.push_back(lund_coords_jet_y);
            lund_kt_events_sd.push_back(lund_kt_jet);
            lund_z_events_sd.push_back(lund_z_jet);
            lund_psi_events_sd.push_back(lund_psi_jet);
            lund_mass_events_sd.push_back(lund_mass_jet);

            double psi1 = lund_psi_jet.front(); //Primary plane psi is just the first declustering's psi

            if (max_kt_index < 0) {
                lund_coords_events_secondary_x_sd.push_back({});
                lund_coords_events_secondary_y_sd.push_back({});
                lund_kt_events_secondary_sd.push_back({});
                lund_z_events_secondary_sd.push_back({});
                lund_psi_events_secondary_sd.push_back({});
                lund_mass_events_secondary_sd.push_back({});

                lund_psi12_events_sd.push_back(numeric_limits<double>::quiet_NaN());
                continue;
            }

            lund_coords_jet_x.clear();
            lund_coords_jet_y.clear();
            lund_kt_jet.clear();
            lund_z_jet.clear();
            lund_psi_jet.clear();
            lund_mass_jet.clear();

            vector<fastjet::contrib::LundDeclustering> declusts_secondary = lund.result(declusters[max_kt_index].softer()); //Secondary plane is the declustering of the softer branch of the first declustering

            if (declusts_secondary.size() == 0) {
                lund_coords_events_secondary_x_sd.push_back({});
                lund_coords_events_secondary_y_sd.push_back({});
                lund_kt_events_secondary_sd.push_back({});
                lund_z_events_secondary_sd.push_back({});
                lund_psi_events_secondary_sd.push_back({});
                lund_mass_events_secondary_sd.push_back({});

                lund_psi12_events_sd.push_back(numeric_limits<double>::quiet_NaN());
                continue;
            }

            for (fastjet::contrib::LundDeclustering declust : declusts_secondary) {
                pair<double,double> coords = declust.lund_coordinates();
                double kt = declust.kt();
                double z = declust.z();
                double psi = declust.psi();
                double mass = declust.m();

                lund_coords_jet_x.push_back(coords.first);
                lund_coords_jet_y.push_back(coords.second);
                lund_kt_jet.push_back(kt);
                lund_z_jet.push_back(z);
                lund_psi_jet.push_back(psi);
                lund_mass_jet.push_back(mass);
            }

            lund_coords_events_secondary_x_sd.push_back(lund_coords_jet_x);
            lund_coords_events_secondary_y_sd.push_back(lund_coords_jet_y);
            lund_kt_events_secondary_sd.push_back(lund_kt_jet);
            lund_z_events_secondary_sd.push_back(lund_z_jet);
            lund_psi_events_secondary_sd.push_back(lund_psi_jet);
            lund_mass_events_secondary_sd.push_back(lund_mass_jet);

            double psi2 = lund_psi_jet.front(); //Secondary plane psi is just the first declustering's psi

            double dpsi12 = psi2 - psi1; //Delta psi12 between primary and secondary planes
            dpsi12 = wrap_pm_pi(dpsi12);
            lund_psi12_events_sd.push_back(dpsi12);
        }

        if (event_count%1000 == 0) {
            cout << "Rank " << rank << ": processed " << event_count << " events of " << nevents << endl;
        }
        event_count++;

        lund_branch_x_sd->Fill();
        lund_branch_y_sd->Fill();
        lund_branch_kt_sd->Fill();
        lund_branch_z_sd->Fill();
        lund_branch_psi_sd->Fill();
        lund_branch_mass_sd->Fill();

        lund_branch_secondary_x_sd->Fill();
        lund_branch_secondary_y_sd->Fill();
        lund_branch_secondary_kt_sd->Fill();
        lund_branch_secondary_z_sd->Fill();
        lund_branch_secondary_psi_sd->Fill();
        lund_branch_secondary_mass_sd->Fill();

        lund_branch_psi12_sd->Fill();

    }
    // Write the tree and close the file
    tree->Write("", TObject::kOverwrite);
    file->Write("", TObject::kOverwrite);
    file->Close();

    cout << "Rank " << rank << ": finished processing " << event_count << " events.\n";

}