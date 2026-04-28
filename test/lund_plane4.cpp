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

//helper function to compute delta phi between two angles (in radians)
struct Vec3 { double x,y,z; };

static inline Vec3 v3(const fastjet::PseudoJet& p){ return {p.px(), p.py(), p.pz()}; }

static inline Vec3 cross(const Vec3& a, const Vec3& b){
  return {a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x};
}

static inline double dot(const Vec3& a, const Vec3& b){ return a.x*b.x + a.y*b.y + a.z*b.z; }

static inline double norm(const Vec3& a){ return std::sqrt(dot(a,a)); }

static inline Vec3 unit(const Vec3& a){
  double n = norm(a);
  if (n==0) return {0,0,0};
  return {a.x/n, a.y/n, a.z/n};
}

// Signed angle between plane normals n_prev -> n_now
// Sign convention: sgn = sign( (n_prev x n_now) · p_hard )
static inline double signed_dpsi(const Vec3& n_prev, const Vec3& n_now, const Vec3& p_hard){
  Vec3 c = cross(n_prev, n_now);
  double ang = std::atan2(norm(c), dot(n_prev, n_now)); // [0,pi]
  if (dot(c, p_hard) < 0) ang = -ang;
  return ang;
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

    auto lund_branch_psi12 = tree->Branch("lund_psi12", &lund_psi12_events);

    // Working vectors for jet-level declustering (reset for each jet)
    vector<double> lund_coords_jet_x;
    vector<double> lund_coords_jet_y;
    vector<double> lund_kt_jet;
    vector<double> lund_z_jet;
    vector<double> lund_psi_jet;
    vector<double> lund_mass_jet;

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

        lund_psi12_events.clear();

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

            if (constituents.size() < 2 ) {

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

                lund_psi12_events.push_back(numeric_limits<double>::quiet_NaN());

                continue;
            }

            // Create Lund generator and Soft Drop objects
            fastjet::contrib::LundGenerator lund;
            fastjet::JetDefinition jet_def(fastjet::cambridge_aachen_algorithm, R0);
            fastjet::ClusterSequence cs(constituents, jet_def);
            auto jets = fastjet::sorted_by_pt(cs.inclusive_jets());

            if (jets.size() == 0 || jets.size() > 1) {
                cout << "Warning: event has " << jets.size() << " jets, skipping" << endl;
            }

            // Get the primary Lund plane nodes
            vector<fastjet::contrib::LundDeclustering> declusts = lund.result(jets[0]);

            bool have_prev = false;
            Vec3 n_prev{0,0,0};
            double psi_acc = 0.0;
            double max_kt = -1;
            vector<double> primary_psi_values;
            int max_kt_index = -1;
            double another_psi1 = numeric_limits<double>::quiet_NaN();

            int i_declust = 0;
            for (fastjet::contrib::LundDeclustering declust : declusts) {
                pair<double,double> coords = declust.lund_coordinates();
                double kt = declust.kt();
                double z = declust.z();
                double psi = declust.psi();
                double mass = declust.m();

                const fastjet::PseudoJet& p_hard = declust.harder();
                const fastjet::PseudoJet& p_soft = declust.softer();
                Vec3 pa = v3(p_hard);
                Vec3 pb = v3(p_soft);
                Vec3 n = unit(cross(pa, pb));
                
                if (!have_prev) {
                    psi_acc = 0.0;
                    n_prev = n;
                    have_prev = true;
                }
                else {
                    psi_acc += signed_dpsi(n_prev, n, pa);
                }
                n_prev = n;
                primary_psi_values.push_back(psi_acc);

                if (kt > max_kt && z > sd_zcut) {
                    max_kt = kt;
                    max_kt_index = i_declust;
                    another_psi1 = psi;
                }

                lund_coords_jet_x.push_back(coords.first);
                lund_coords_jet_y.push_back(coords.second);
                lund_kt_jet.push_back(kt);
                lund_z_jet.push_back(z);
                lund_psi_jet.push_back(psi);
                lund_mass_jet.push_back(mass);

                i_declust++;
            }

            lund_coords_events_x.push_back(lund_coords_jet_x);
            lund_coords_events_y.push_back(lund_coords_jet_y);
            lund_kt_events.push_back(lund_kt_jet);
            lund_z_events.push_back(lund_z_jet);
            lund_psi_events.push_back(lund_psi_jet);
            lund_mass_events.push_back(lund_mass_jet);

            lund_coords_jet_x.clear();
            lund_coords_jet_y.clear();
            lund_kt_jet.clear();
            lund_z_jet.clear();
            lund_psi_jet.clear();
            lund_mass_jet.clear();

            double psi1 = std::numeric_limits<double>::quiet_NaN();
            if (max_kt_index < 0){

                lund_coords_events_secondary_x.push_back({});
                lund_coords_events_secondary_y.push_back({});
                lund_kt_events_secondary.push_back({});
                lund_z_events_secondary.push_back({});
                lund_psi_events_secondary.push_back({});
                lund_mass_events_secondary.push_back({});

                lund_psi12_events.push_back(numeric_limits<double>::quiet_NaN());

                continue;

            }
            
            psi1 = primary_psi_values[max_kt_index];

            vector<fastjet::contrib::LundDeclustering> declusts2 = lund.result(declusts[max_kt_index].softer());
            
            // compute normal for the chosen primary splitting
            const auto& prim = declusts[max_kt_index];
            Vec3 n_prim = unit(cross(v3(prim.harder()), v3(prim.softer())));

            // seed secondary chain from the primary splitting
            Vec3 n_prev2 = n_prim;
            double psi_acc2 = primary_psi_values[max_kt_index]; // start from the psi value of the primary splitting
            bool have_prev2 = true; // because n_prev2 is already set

            double max_kt2 = -1;
            vector<double> secondary_psi_values;
            int max_kt_index2 = -1;
            double another_psi2 = numeric_limits<double>::quiet_NaN();

            int i_declust2 = 0;
            for (fastjet::contrib::LundDeclustering declust : declusts2) {
                pair<double,double> coords = declust.lund_coordinates();
                double kt = declust.kt();
                double z = declust.z();
                double psi = declust.psi();
                double mass = declust.m();

                const fastjet::PseudoJet& p_hard = declust.harder();
                const fastjet::PseudoJet& p_soft = declust.softer();
                Vec3 pa = v3(p_hard);
                Vec3 pb = v3(p_soft);
                Vec3 n = unit(cross(pa, pb));

                if (!have_prev2) {
                    psi_acc2 = 0.0;
                    have_prev2 = true;
                }
                else {
                    psi_acc2 += signed_dpsi(n_prev2, n, pa);
                }
                n_prev2 = n;
                secondary_psi_values.push_back(psi_acc2);

                if (kt > max_kt2 && z > sd_zcut) {
                    max_kt2 = kt;
                    max_kt_index2 = i_declust2;
                    another_psi2 = psi;
                }

                lund_coords_jet_x.push_back(coords.first);
                lund_coords_jet_y.push_back(coords.second);
                lund_kt_jet.push_back(kt);
                lund_z_jet.push_back(z);
                lund_psi_jet.push_back(psi);
                lund_mass_jet.push_back(mass);

                i_declust2++;
            }

            double psi2 = std::numeric_limits<double>::quiet_NaN();
            if (max_kt_index2 < 0){
                lund_psi12_events.push_back(numeric_limits<double>::quiet_NaN());
            } else {
                psi2 = secondary_psi_values[max_kt_index2];
                double dpsi12 = another_psi2 - another_psi1; // Use the psi values of the max kt splittings for delta psi12
                dpsi12 = wrap_pm_pi(dpsi12);
                lund_psi12_events.push_back(dpsi12);
                //cout << "Event " << event_count << ": psi1 = " << psi1 << ", psi2 = " << psi2 << ", delta_psi12 = " << (psi2 - psi1) << endl;
            }

            lund_coords_events_secondary_x.push_back(lund_coords_jet_x);
            lund_coords_events_secondary_y.push_back(lund_coords_jet_y);
            lund_kt_events_secondary.push_back(lund_kt_jet);
            lund_z_events_secondary.push_back(lund_z_jet);
            lund_psi_events_secondary.push_back(lund_psi_jet);
            lund_mass_events_secondary.push_back(lund_mass_jet);

            lund_coords_jet_x.clear();
            lund_coords_jet_y.clear();
            lund_kt_jet.clear();
            lund_z_jet.clear();
            lund_psi_jet.clear();
            lund_mass_jet.clear();
        }

        if (event_count%1000 == 0) {
            cout << "Rank " << rank << ": processed " << event_count << " events of " << nevents << endl;
        }
        event_count++;

        // Fill the branches for this event
        lund_branch_x ->Fill();
        lund_branch_y ->Fill();
        lund_branch_kt ->Fill();
        lund_branch_z ->Fill();
        lund_branch_psi ->Fill();
        lund_branch_mass ->Fill();

        lund_branch_secondary_x ->Fill();
        lund_branch_secondary_y ->Fill();
        lund_branch_secondary_kt ->Fill();
        lund_branch_secondary_z ->Fill();
        lund_branch_secondary_psi ->Fill();
        lund_branch_secondary_mass ->Fill();

        lund_branch_psi12 ->Fill();
    }
    
    // Write the tree and close the file
    tree->Write("", TObject::kOverwrite);
    file->Write("", TObject::kOverwrite);
    file->Close();

    cout << "Rank " << rank << ": finished processing " << event_count << " events.\n";
}