/*
g++ -std=c++17 -O2 \
main/lund_plane3.cpp -o build/lund_plane3 \
$(fastjet-config --cxxflags) $(root-config --cflags) \
$(fastjet-config --libs) -lfastjetplugins -lfastjettools -lfastjetcontribfragile \
$(root-config --libs)
*/

//FastJet includes
#include "fastjet/PseudoJet.hh"
#include "fastjet/EECambridgePlugin.hh"
#include "fastjet/contrib/RecursiveLundEEGenerator.hh"
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

using namespace std;

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
    int depth = -1;

    int event_count = 0;

    // Create ROOT branches for output
    vector< vector<double> > lund_coords_events_x; //Primary plane coordinates
    vector< vector<double> > lund_coords_events_y;
    vector< vector<double> > lund_kt_events;
    vector< vector<double> > lund_z_events;
    vector< vector<double> > lund_psi_events;
    vector< vector<double> > lund_mass_events;

    vector< vector<double> > lund_coords_events_secondary_x; //Secondary plane coordinates
    vector< vector<double> > lund_coords_events_secondary_y;
    vector< vector<double> > lund_kt_events_secondary;
    vector< vector<double> > lund_z_events_secondary;
    vector< vector<double> > lund_psi_events_secondary;
    vector< vector<double> > lund_mass_events_secondary;

    vector< vector<double> > lund_coords_events_x_sd; //Soft drop primary plane coordinates
    vector< vector<double> > lund_coords_events_y_sd;
    vector< vector<double> > lund_kt_events_sd;
    vector< vector<double> > lund_z_events_sd;
    vector< vector<double> > lund_psi_events_sd;
    vector< vector<double> > lund_mass_events_sd;

    vector< vector<double> > lund_coords_events_secondary_x_sd; //Soft drop secondary plane coordinates
    vector< vector<double> > lund_coords_events_secondary_y_sd;
    vector< vector<double> > lund_kt_events_secondary_sd;
    vector< vector<double> > lund_z_events_secondary_sd;
    vector< vector<double> > lund_psi_events_secondary_sd;
    vector< vector<double> > lund_mass_events_secondary_sd;

    vector< double> lund_psi12_events; //Delta psi12 between primary and secondary planes

    // Setup ROOT branches
    auto lund_branch_x = tree->Branch("lund_coords_x", &lund_coords_events_x);
    auto lund_branch_y = tree->Branch("lund_coords_y", &lund_coords_events_y);
    auto lund_branch_kt = tree->Branch("lund_kt", &lund_kt_events);
    auto lund_branch_z = tree->Branch("lund_z", &lund_z_events);
    auto lund_branch_psi = tree->Branch("lund_psi", &lund_psi_events);
    auto lund_branch_mass = tree->Branch("lund_mass", &lund_mass_events);

    auto lund_branch_secondary_x = tree->Branch("lund_coords_secondary_x", &lund_coords_events_secondary_x);
    auto lund_branch_secondary_y = tree->Branch("lund_coords_secondary_y", &lund_coords_events_secondary_y);
    auto lund_branch_secondary_kt = tree->Branch("lund_kt_secondary", &lund_kt_events_secondary);
    auto lund_branch_secondary_z = tree->Branch("lund_z_secondary", &lund_z_events_secondary);
    auto lund_branch_secondary_psi = tree->Branch("lund_psi_secondary", &lund_psi_events_secondary);
    auto lund_branch_secondary_mass = tree->Branch("lund_mass_secondary", &lund_mass_events_secondary);

    auto lund_branch_x_sd = tree->Branch("lund_coords_x_sd", &lund_coords_events_x_sd);
    auto lund_branch_y_sd = tree->Branch("lund_coords_y_sd", &lund_coords_events_y_sd);
    auto lund_branch_kt_sd = tree->Branch("lund_kt_sd", &lund_kt_events_sd);
    auto lund_branch_z_sd = tree->Branch("lund_z_sd", &lund_z_events_sd);
    auto lund_branch_psi_sd = tree->Branch("lund_psi_sd", &lund_psi_events_sd);
    auto lund_branch_mass_sd = tree->Branch("lund_mass_sd", &lund_mass_events_sd);

    auto lund_branch_x_sd_secondary = tree->Branch("lund_coords_x_sd_secondary", &lund_coords_events_secondary_x_sd);
    auto lund_branch_y_sd_secondary = tree->Branch("lund_coords_y_sd_secondary", &lund_coords_events_secondary_y_sd);
    auto lund_branch_kt_sd_secondary = tree->Branch("lund_kt_sd_secondary", &lund_kt_events_secondary_sd);
    auto lund_branch_z_sd_secondary = tree->Branch("lund_z_sd_secondary", &lund_z_events_secondary_sd);
    auto lund_branch_psi_sd_secondary = tree->Branch("lund_psi_sd_secondary", &lund_psi_events_secondary_sd);
    auto lund_branch_mass_sd_secondary = tree->Branch("lund_mass_sd_secondary", &lund_mass_events_secondary_sd);

    auto lund_branch_psi12 = tree->Branch("lund_psi12", &lund_psi12_events);

    // Working vectors for jet-level declustering (reset for each jet)
    vector<double> lund_coords_jet_x;
    vector<double> lund_coords_jet_y;
    vector<double> lund_kt_jet;
    vector<double> lund_z_jet;
    vector<double> lund_psi_jet;
    vector<double> lund_mass_jet;

    vector<double> lund_coords_jet_x_sd;
    vector<double> lund_coords_jet_y_sd;
    vector<double> lund_kt_jet_sd;
    vector<double> lund_z_jet_sd;
    vector<double> lund_psi_jet_sd;
    vector<double> lund_mass_jet_sd;

    while (reader.Next()){
        // Clear branches
        lund_coords_events_x.clear();
        lund_coords_events_y.clear();
        lund_kt_events.clear();
        lund_z_events.clear();
        lund_psi_events.clear();
        lund_mass_events.clear();

        lund_coords_events_secondary_x.clear();
        lund_coords_events_secondary_y.clear();
        lund_kt_events_secondary.clear();
        lund_z_events_secondary.clear();
        lund_psi_events_secondary.clear();
        lund_mass_events_secondary.clear();

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

        lund_psi12_events.clear();

        for (size_t ijet=0; ijet < jet_pt->size(); ++ijet) {

            lund_coords_jet_x.clear();
            lund_coords_jet_y.clear();
            lund_kt_jet.clear();
            lund_z_jet.clear();
            lund_psi_jet.clear();
            lund_mass_jet.clear();
            
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

            // If fewer than 2 particles, no declustering possible: push empty entries
            
            if (constituents.size() < 2) {
                lund_coords_events_x.push_back({});
                lund_coords_events_y.push_back({});
                lund_kt_events.push_back({});
                lund_z_events.push_back({});
                lund_psi_events.push_back({});
                lund_mass_events.push_back({});

                lund_coords_events_secondary_x.push_back({});
                lund_coords_events_secondary_y.push_back({});
                lund_kt_events_secondary.push_back({});
                lund_z_events_secondary.push_back({});
                lund_psi_events_secondary.push_back({});
                lund_mass_events_secondary.push_back({});

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

                // optional: keep psi12 aligned; choose a sentinel
                lund_psi12_events.push_back(numeric_limits<double>::quiet_NaN());
                continue;
            }   

            //fastjet::JetDefinition jet_def(fastjet::cambridge_aachen_algorithm, R0);
            fastjet::JetDefinition::Plugin* ee_plugin = new fastjet::EECambridgePlugin(R0);
            fastjet::JetDefinition jet_def(ee_plugin);
            fastjet::contrib::RecursiveLundEEGenerator lund(depth);
            fastjet::contrib::SoftDrop softdrop(sd_beta, sd_zcut, R0);

            // Find the two highest-kt primary declustering (ordered in kt by default)
            fastjet::ClusterSequence cs(constituents, jet_def);
            vector<fastjet::contrib::LundEEDeclustering> declusts = lund.result(cs);
            double psi_primary_1;

            int i_primary_1 = -1;
            double best_kt_primary = -1;

            for (unsigned int i=0; i<declusts.size(); ++i) {
                if (declusts[i].depth() == 0){
                    pair<double,double> coords = declusts[i].lund_coordinates();
                    double kt = declusts[i].kt();
                    double z = declusts[i].z();
                    double psi = declusts[i].psi();
                    double mass = declusts[i].m();

                    lund_coords_jet_x.push_back(coords.first);
                    lund_coords_jet_y.push_back(coords.second);
                    lund_kt_jet.push_back(kt);
                    lund_z_jet.push_back(z);
                    lund_psi_jet.push_back(psi);
                    lund_mass_jet.push_back(mass);

                    if (declusts[i].z() > sd_zcut){
                        if(i_primary_1 < 0) {
                            i_primary_1 = i;
                            psi_primary_1 = declusts[i].psibar();
                        }
                    
                        lund_coords_jet_x_sd.push_back(coords.first);
                        lund_coords_jet_y_sd.push_back(coords.second);
                        lund_kt_jet_sd.push_back(kt);
                        lund_z_jet_sd.push_back(z);
                        lund_psi_jet_sd.push_back(psi);
                        lund_mass_jet_sd.push_back(mass);   
                    }
                }
            }

            //Add to event-level vectors
            lund_coords_events_x.push_back(lund_coords_jet_x);
            lund_coords_events_y.push_back(lund_coords_jet_y);
            lund_kt_events.push_back(lund_kt_jet);
            lund_z_events.push_back(lund_z_jet);
            lund_psi_events.push_back(lund_psi_jet);
            lund_mass_events.push_back(lund_mass_jet);

            lund_coords_events_x_sd.push_back(lund_coords_jet_x_sd);
            lund_coords_events_y_sd.push_back(lund_coords_jet_y_sd);
            lund_kt_events_sd.push_back(lund_kt_jet_sd);
            lund_z_events_sd.push_back(lund_z_jet_sd);
            lund_psi_events_sd.push_back(lund_psi_jet_sd);
            lund_mass_events_sd.push_back(lund_mass_jet_sd);

            // Reset vectors for secondary declustering
            lund_coords_jet_x.clear();
            lund_coords_jet_y.clear();
            lund_kt_jet.clear();
            lund_z_jet.clear();
            lund_psi_jet.clear();
            lund_mass_jet.clear();

            lund_coords_jet_x_sd.clear();
            lund_coords_jet_y_sd.clear();
            lund_kt_jet_sd.clear();
            lund_z_jet_sd.clear();
            lund_psi_jet_sd.clear();
            lund_mass_jet_sd.clear();

            // If no primary passing zcut: still need to push empty secondary + psi12 sentinel
            if (i_primary_1 < 0) {
                lund_coords_events_secondary_x.push_back({});
                lund_coords_events_secondary_y.push_back({});
                lund_kt_events_secondary.push_back({});
                lund_z_events_secondary.push_back({});
                lund_psi_events_secondary.push_back({});
                lund_mass_events_secondary.push_back({});

                lund_coords_events_secondary_x_sd.push_back({});
                lund_coords_events_secondary_y_sd.push_back({});
                lund_kt_events_secondary_sd.push_back({});
                lund_z_events_secondary_sd.push_back({});
                lund_psi_events_secondary_sd.push_back({});
                lund_mass_events_secondary_sd.push_back({});

                lund_psi12_events.push_back(std::numeric_limits<double>::quiet_NaN());
                continue;
            }

            // For the highest-kt primary: find the highest-kt secondary associated to that Lund leaf
            // that passes the zcut of z2_cut
            int iplane_to_follow = declusts[i_primary_1].leaf_iplane();
            vector<const fastjet::contrib::LundEEDeclustering *> secondaries;
            for (const auto & declust: declusts){
                if (declust.iplane() == iplane_to_follow) secondaries.push_back(&declust);
            }

            int i_secondary = -1;
            double dpsi_12 = numeric_limits<double>::max(); //< initialisation prevents gcc warning;

            bool found_psi12 = false;

            if (secondaries.size()>0){
                for (unsigned int i=0; i<secondaries.size(); ++i) {
                    pair<double,double> coords = secondaries[i]->lund_coordinates();
                    double kt = secondaries[i]->kt();
                    double z = secondaries[i]->z();
                    double psi = secondaries[i]->psi();
                    double mass = secondaries[i]->m();

                    lund_coords_jet_x.push_back(coords.first);
                    lund_coords_jet_y.push_back(coords.second);
                    lund_kt_jet.push_back(kt);
                    lund_z_jet.push_back(z);
                    lund_psi_jet.push_back(psi);
                    lund_mass_jet.push_back(mass);

                    if (secondaries[i]->z() > sd_zcut) {
                        if (!found_psi12) {
                        i_secondary = (int)i;
                        double psi_2 = secondaries[i]->psibar();
                        dpsi_12 = fastjet::contrib::lund_plane::map_to_pi(psi_2 - psi_primary_1);
                        lund_psi12_events.push_back(dpsi_12);
                        found_psi12 = true;
                        }

                        lund_coords_jet_x_sd.push_back(coords.first);
                        lund_coords_jet_y_sd.push_back(coords.second);
                        lund_kt_jet_sd.push_back(kt);
                        lund_z_jet_sd.push_back(z);
                        lund_psi_jet_sd.push_back(psi);
                        lund_mass_jet_sd.push_back(mass);
                    }
                }

                if (!found_psi12) {
                    lund_psi12_events.push_back(numeric_limits<double>::quiet_NaN());
                }
            } else {
                // No secondaries at all
                lund_coords_events_secondary_x.push_back({});
                lund_coords_events_secondary_y.push_back({});
                lund_kt_events_secondary.push_back({});
                lund_z_events_secondary.push_back({});
                lund_psi_events_secondary.push_back({});
                lund_mass_events_secondary.push_back({});

                lund_coords_events_secondary_x_sd.push_back({});
                lund_coords_events_secondary_y_sd.push_back({});
                lund_kt_events_secondary_sd.push_back({});
                lund_z_events_secondary_sd.push_back({});
                lund_psi_events_secondary_sd.push_back({});
                lund_mass_events_secondary_sd.push_back({});

                lund_psi12_events.push_back(numeric_limits<double>::quiet_NaN());
                continue;
            }

            //Add to event-level vectors
            lund_coords_events_secondary_x.push_back(lund_coords_jet_x);
            lund_coords_events_secondary_y.push_back(lund_coords_jet_y);
            lund_kt_events_secondary.push_back(lund_kt_jet);
            lund_z_events_secondary.push_back(lund_z_jet);
            lund_psi_events_secondary.push_back(lund_psi_jet);
            lund_mass_events_secondary.push_back(lund_mass_jet);

            lund_coords_events_secondary_x_sd.push_back(lund_coords_jet_x_sd);
            lund_coords_events_secondary_y_sd.push_back(lund_coords_jet_y_sd);
            lund_kt_events_secondary_sd.push_back(lund_kt_jet_sd);
            lund_z_events_secondary_sd.push_back(lund_z_jet_sd);
            lund_psi_events_secondary_sd.push_back(lund_psi_jet_sd);
            lund_mass_events_secondary_sd.push_back(lund_mass_jet_sd);

            //Reset vectors for soft drop primary declustering
            lund_coords_jet_x.clear();
            lund_coords_jet_y.clear();
            lund_kt_jet.clear();
            lund_z_jet.clear();
            lund_psi_jet.clear();
            lund_mass_jet.clear();

            lund_coords_jet_x_sd.clear();
            lund_coords_jet_y_sd.clear();
            lund_kt_jet_sd.clear();
            lund_z_jet_sd.clear();
            lund_psi_jet_sd.clear();
            lund_mass_jet_sd.clear();
        }

        if (event_count%1000 == 0) {
            cout << "Rank " << rank << ": processed " << event_count << " events of " << nevents << endl;
        }
        event_count++;
        // Fill the branches for this event
        lund_branch_x->Fill();
        lund_branch_y->Fill();
        lund_branch_kt->Fill();
        lund_branch_z->Fill();
        lund_branch_psi->Fill();
        lund_branch_mass->Fill();
        
        lund_branch_secondary_x->Fill();
        lund_branch_secondary_y->Fill();
        lund_branch_secondary_kt->Fill();
        lund_branch_secondary_z->Fill();
        lund_branch_secondary_psi->Fill();
        lund_branch_secondary_mass->Fill();

        lund_branch_x_sd->Fill();
        lund_branch_y_sd->Fill();
        lund_branch_kt_sd->Fill();
        lund_branch_z_sd->Fill();
        lund_branch_psi_sd->Fill();
        lund_branch_mass_sd->Fill();

        lund_branch_x_sd_secondary->Fill();
        lund_branch_y_sd_secondary->Fill();
        lund_branch_kt_sd_secondary->Fill();
        lund_branch_z_sd_secondary->Fill();
        lund_branch_psi_sd_secondary->Fill();
        lund_branch_mass_sd_secondary->Fill();

        lund_branch_psi12->Fill();

    }

    tree->Write("", TObject::kOverwrite);
    file->Write("", TObject::kOverwrite);
    file->Close();

    cout << "Rank " << rank << ": finished processing " << event_count << " events.\n";

}