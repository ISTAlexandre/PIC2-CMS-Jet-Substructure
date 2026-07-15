/*
g++ -std=c++17 -O2 \
main/lund_softdrop2.cpp -o build/lund_softdrop2 \
$(fastjet-config --cxxflags) $(root-config --cflags) \
$(fastjet-config --libs) -lfastjetplugins -lfastjettools -lfastjetcontribfragile \
$(root-config --libs)
*/

//FastJet includes
#include "fastjet/PseudoJet.hh"
#include "fastjet/EECambridgePlugin.hh"
#include "fastjet/contrib/LundGenerator.hh"

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
    x = fmod(x + M_PI, two_pi);
    if (x < 0) x += two_pi;
    x -= M_PI;
  
    // Make endpoint convention (-pi, pi]
    if (x <= -M_PI) x += two_pi;
    return x;
}

static inline vector<double> plane_normal(const fastjet::PseudoJet& p1, const fastjet::PseudoJet& p2) {
    // Compute the normal vector to the plane defined by p1 and p2
    double px = p1.py()*p2.pz() - p1.pz()*p2.py();
    double py = p1.pz()*p2.px() - p1.px()*p2.pz();
    double pz = p1.px()*p2.py() - p1.py()*p2.px();
    double norm = sqrt(px*px + py*py + pz*pz);
    if (norm > 0) {
    return {px/norm, py/norm, pz/norm};
    } else {
    return {0.0, 0.0, 0.0}; // Degenerate case
    }
}

static inline double dot_product(const vector<double>& v1, const vector<double>& v2) {
    return v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2];
}

static inline vector<double> cross_product(const vector<double>& v1, const vector<double>& v2) {
    return {v1[1]*v2[2] - v1[2]*v2[1], v1[2]*v2[0] - v1[0]*v2[2], v1[0]*v2[1] - v1[1]*v2[0]};
}

static inline int sign(double x) {
    if (x > 0) return 1;
    else if (x < 0) return -1;
    else return 0;
}

static inline double angle_between_planes(fastjet::PseudoJet p1, fastjet::PseudoJet p2, fastjet::PseudoJet p3, fastjet::PseudoJet p4) {
    // Compute the angle between the planes defined by (p1,p2) and (p3,p4)
    vector<double> n1 = plane_normal(p1, p2);
    vector<double> n2 = plane_normal(p3, p4);
    double dot = dot_product(n1, n2);

    vector<double> cross_prod = cross_product(n1,n2);
    double sin_ang = sign(dot_product(cross_prod, {p1.px(), p1.py(), p1.pz()}));

    double cos = dot * sin_ang;
    double angle = acos(cos);
    return angle;
}

static inline double cms_delta_phi(const fastjet::PseudoJet& p1, const fastjet::PseudoJet& p2, const fastjet::PseudoJet& p3, const fastjet::PseudoJet& p4) {
    auto n1 = plane_normal(p1, p2);
    auto n2 = plane_normal(p3, p4);
    if (n1 == vector<double>{0.0, 0.0, 0.0} || n2 == vector<double>{0.0, 0.0, 0.0}) {
        // Degenerate case where one of the planes is not well-defined
        return numeric_limits<double>::quiet_NaN();
    }

    double c = std::clamp(dot_product(n1, n2), -1.0, 1.0);

    auto cp = cross_product(n1, n2);

    // CMS uses the most energetic of partons 1 and 2.
    const fastjet::PseudoJet& phard = (p1.E() > p2.E()) ? p1 : p2;

    double orient = cp[0]*phard.px() + cp[1]*phard.py() + cp[2]*phard.pz();
    double sgn = (orient >= 0.0) ? 1.0 : -1.0;

    // This is the published CMS-style angle in [0, pi]
    return std::acos(std::clamp(c * sgn, -1.0, 1.0));
}

static inline double cms_delta_phi_signed(const fastjet::PseudoJet& p1, const fastjet::PseudoJet& p2, const fastjet::PseudoJet& p3, const fastjet::PseudoJet& p4) {
    auto n1 = plane_normal(p1, p2);
    auto n2 = plane_normal(p3, p4);

    double c = std::clamp(dot_product(n1, n2), -1.0, 1.0);

    auto cp = cross_product(n1, n2);

    const fastjet::PseudoJet& phard = (p1.E() > p2.E()) ? p1 : p2;

    double orient = cp[0]*phard.px() + cp[1]*phard.py() + cp[2]*phard.pz();
    double sgn = (orient >= 0.0) ? 1.0 : -1.0;

    double dphi_cms = std::acos(std::clamp(c * sgn, -1.0, 1.0));  // [0, pi]
    return wrap_pm_pi(sgn * dphi_cms);                            // (-pi, pi]
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
    const double sd_beta_secondary = 0; // beta for secondary plane declustering (if different from primary)
    const double sd_zcut = 0.0;     // typical 0.05–0.2
    const double sd_zcut_secondary = 0.0; // beta for secondary plane declustering (if different from primary)
    const double R0   = 1.0;     // usually = jet R
    const double soft_min_pt = 130.0; // Minimum pT for softer branch to pass soft drop condition (CMS-style)

    int event_count = 0;

    // Create ROOT branches for output
    vector< vector<double> > lund_coords_events_x_sd; //Primary plane coordinates
    vector< vector<double> > lund_coords_events_y_sd;
    vector< vector<double> > lund_kt_events_sd;
    vector< vector<double> > lund_z_events_sd;
    vector< vector<double> > lund_psi_events_sd;
    vector< vector<double> > lund_delta_events_sd;
    vector< vector<double> > lund_mass_events_sd;

    vector< vector<double> > lund_coords_events_secondary_x_sd; //Secondary plane coordinates
    vector< vector<double> > lund_coords_events_secondary_y_sd;
    vector< vector<double> > lund_kt_events_secondary_sd;
    vector< vector<double> > lund_z_events_secondary_sd;
    vector< vector<double> > lund_psi_events_secondary_sd;
    vector< vector<double> > lund_delta_events_secondary_sd;
    vector< vector<double> > lund_mass_events_secondary_sd;

    vector< double> lund_psi12_events_sd; //Delta psi12 between primary and secondary planes
    vector< int> lund_max_kt_events_sd; //Max kT of declusterings in primary plane
    vector< int> lund_max_kt_secondary_events_sd; //Max kT of declusterings in secondary plane

    //Setup ROOT branches
    auto lund_branch_x_sd = tree->Branch("lund_coords_x_sd", &lund_coords_events_x_sd);
    auto lund_branch_y_sd = tree->Branch("lund_coords_y_sd", &lund_coords_events_y_sd);
    auto lund_branch_kt_sd = tree->Branch("lund_kt_sd", &lund_kt_events_sd);
    auto lund_branch_z_sd = tree->Branch("lund_z_sd", &lund_z_events_sd);
    auto lund_branch_psi_sd = tree->Branch("lund_psi_sd", &lund_psi_events_sd);
    auto lund_branch_delta_sd = tree->Branch("lund_delta_sd", &lund_delta_events_sd);
    auto lund_branch_mass_sd = tree->Branch("lund_mass_sd", &lund_mass_events_sd);
    
    auto lund_branch_secondary_x_sd = tree->Branch("lund_coords_secondary_x_sd", &lund_coords_events_secondary_x_sd);
    auto lund_branch_secondary_y_sd = tree->Branch("lund_coords_secondary_y_sd", &lund_coords_events_secondary_y_sd);
    auto lund_branch_secondary_kt_sd = tree->Branch("lund_kt_secondary_sd", &lund_kt_events_secondary_sd);
    auto lund_branch_secondary_z_sd = tree->Branch("lund_z_secondary_sd", &lund_z_events_secondary_sd);
    auto lund_branch_secondary_psi_sd = tree->Branch("lund_psi_secondary_sd", &lund_psi_events_secondary_sd);
    auto lund_branch_secondary_delta_sd = tree->Branch("lund_delta_secondary_sd", &lund_delta_events_secondary_sd);
    auto lund_branch_secondary_mass_sd = tree->Branch("lund_mass_secondary_sd", &lund_mass_events_secondary_sd);

    auto lund_branch_psi12_sd = tree->Branch("lund_psi12_sd", &lund_psi12_events_sd);
    auto lund_branch_max_kt_sd = tree->Branch("lund_max_kt_sd", &lund_max_kt_events_sd);
    auto lund_branch_max_kt_secondary_sd = tree->Branch("lund_max_kt_secondary_sd", &lund_max_kt_secondary_events_sd);

    // Working vectors for jet-level declustering (reset for each jet)
    vector<double> lund_coords_jet_x;
    vector<double> lund_coords_jet_y;
    vector<double> lund_kt_jet;
    vector<double> lund_z_jet;
    vector<double> lund_psi_jet;
    vector<double> lund_delta_jet;
    vector<double> lund_mass_jet;

    while (reader.Next()){
        //Clear branches
        lund_coords_events_x_sd.clear();
        lund_coords_events_y_sd.clear();
        lund_kt_events_sd.clear();
        lund_z_events_sd.clear();
        lund_psi_events_sd.clear();
        lund_delta_events_sd.clear();
        lund_mass_events_sd.clear();

        lund_coords_events_secondary_x_sd.clear();
        lund_coords_events_secondary_y_sd.clear();
        lund_kt_events_secondary_sd.clear();
        lund_z_events_secondary_sd.clear();
        lund_psi_events_secondary_sd.clear();
        lund_delta_events_secondary_sd.clear();
        lund_mass_events_secondary_sd.clear();

        lund_psi12_events_sd.clear();
        lund_max_kt_events_sd.clear();
        lund_max_kt_secondary_events_sd.clear();

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
            lund_delta_jet.clear();
            lund_mass_jet.clear();

            if (constituents.size() < 2) {
                // Not enough constituents to decluster
                lund_coords_events_x_sd.push_back({});
                lund_coords_events_y_sd.push_back({});
                lund_kt_events_sd.push_back({});
                lund_z_events_sd.push_back({});
                lund_psi_events_sd.push_back({});
                lund_delta_events_sd.push_back({});
                lund_mass_events_sd.push_back({});

                lund_coords_events_secondary_x_sd.push_back({});
                lund_coords_events_secondary_y_sd.push_back({});
                lund_kt_events_secondary_sd.push_back({});
                lund_z_events_secondary_sd.push_back({});
                lund_psi_events_secondary_sd.push_back({});
                lund_delta_events_secondary_sd.push_back({});
                lund_mass_events_secondary_sd.push_back({});

                lund_psi12_events_sd.push_back(numeric_limits<double>::quiet_NaN());
                lund_max_kt_events_sd.push_back(-1);
                lund_max_kt_secondary_events_sd.push_back(-1);
                continue;
            }

            //Create lund generator and soft drop declusterer
            fastjet::contrib::LundGenerator lund;
            fastjet::JetDefinition jet_def(fastjet::cambridge_aachen_algorithm, R0);
            fastjet::ClusterSequence cs(constituents, jet_def);
            auto jets = fastjet::sorted_by_pt(cs.inclusive_jets());

            vector<fastjet::contrib::LundDeclustering> declusters = lund.result(jets[0]);

            double max_kt = -1.0;
            int max_kt_index = -1;
            int i_declust = 0;
            int n_passed_sd = 0;
            int max_declust_index = -1;
            for (fastjet::contrib::LundDeclustering declust : declusters) {
                double z = declust.z();
                double delta = declust.Delta();
                fastjet::PseudoJet soft = declust.softer();
                double soft_pt = soft.pt();
                bool passes = (z > sd_zcut * pow(delta/R0, sd_beta) && soft_pt > soft_min_pt);

                if (passes) {
                    pair<double,double> coords = declust.lund_coordinates();
                    double kt = declust.kt();
                    double psi = declust.psi();
                    double mass = declust.m();

                    if (kt > max_kt) {
                        max_kt = kt;
                        max_kt_index = n_passed_sd;
                        max_declust_index = i_declust;
                    }
                    n_passed_sd++;

                    lund_coords_jet_x.push_back(coords.first);
                    lund_coords_jet_y.push_back(coords.second);
                    lund_kt_jet.push_back(kt);
                    lund_z_jet.push_back(z);
                    lund_psi_jet.push_back(psi);
                    lund_delta_jet.push_back(delta);
                    lund_mass_jet.push_back(mass);
                }
                i_declust++;
            }
            
            if (max_kt_index < 0) {
                // No declusterings passed soft drop
                lund_coords_events_x_sd.push_back({});
                lund_coords_events_y_sd.push_back({});
                lund_kt_events_sd.push_back({});
                lund_z_events_sd.push_back({});
                lund_psi_events_sd.push_back({});
                lund_delta_events_sd.push_back({});
                lund_mass_events_sd.push_back({});

                lund_coords_events_secondary_x_sd.push_back({});
                lund_coords_events_secondary_y_sd.push_back({});
                lund_kt_events_secondary_sd.push_back({});
                lund_z_events_secondary_sd.push_back({});
                lund_psi_events_secondary_sd.push_back({});
                lund_delta_events_secondary_sd.push_back({});
                lund_mass_events_secondary_sd.push_back({});

                lund_psi12_events_sd.push_back(numeric_limits<double>::quiet_NaN());
                lund_max_kt_events_sd.push_back(-1);
                lund_max_kt_secondary_events_sd.push_back(-1);
                continue;
            }
            
            //double psi1 = declusters[max_declust_index].psi();
            fastjet::PseudoJet p1 = declusters[max_declust_index].harder();
            fastjet::PseudoJet p2 = declusters[max_declust_index].softer();
            lund_max_kt_events_sd.push_back(max_kt_index);
            
            lund_coords_events_x_sd.push_back(lund_coords_jet_x);
            lund_coords_events_y_sd.push_back(lund_coords_jet_y);
            lund_kt_events_sd.push_back(lund_kt_jet);
            lund_z_events_sd.push_back(lund_z_jet);
            lund_psi_events_sd.push_back(lund_psi_jet);
            lund_delta_events_sd.push_back(lund_delta_jet);
            lund_mass_events_sd.push_back(lund_mass_jet);

            lund_coords_jet_x.clear();
            lund_coords_jet_y.clear();
            lund_kt_jet.clear();
            lund_z_jet.clear();
            lund_psi_jet.clear();
            lund_delta_jet.clear();
            lund_mass_jet.clear();

            auto groomed_softer = declusters[max_declust_index].softer();
            auto declusts_secondary = lund.result(groomed_softer);

            if (declusts_secondary.size() == 0) {
                lund_coords_events_secondary_x_sd.push_back({});
                lund_coords_events_secondary_y_sd.push_back({});
                lund_kt_events_secondary_sd.push_back({});
                lund_z_events_secondary_sd.push_back({});
                lund_psi_events_secondary_sd.push_back({});
                lund_delta_events_secondary_sd.push_back({});
                lund_mass_events_secondary_sd.push_back({});

                lund_psi12_events_sd.push_back(numeric_limits<double>::quiet_NaN());
                lund_max_kt_secondary_events_sd.push_back(-1);
                continue;
            }

            max_kt = -1.0;
            max_kt_index = -1;
            i_declust = 0;
            n_passed_sd = 0;
            max_declust_index = -1;
            for (fastjet::contrib::LundDeclustering declust : declusts_secondary) {
                double z = declust.z();
                double delta = declust.Delta();
                bool passes = (z > sd_zcut_secondary * pow(delta/R0, sd_beta_secondary));

                if (passes) {
                    pair<double,double> coords = declust.lund_coordinates();
                    double kt = declust.kt();
                    double psi = declust.psi();
                    double mass = declust.m();

                    if (kt > max_kt) {
                        max_kt = kt;
                        max_kt_index = n_passed_sd;
                        max_declust_index = i_declust;
                    }
                    n_passed_sd++;
                    
                    lund_coords_jet_x.push_back(coords.first);
                    lund_coords_jet_y.push_back(coords.second);
                    lund_kt_jet.push_back(kt);
                    lund_z_jet.push_back(z);
                    lund_psi_jet.push_back(psi);
                    lund_delta_jet.push_back(delta);
                    lund_mass_jet.push_back(mass);
                }
                i_declust++;
            }
            
            if (max_kt_index < 0) {
                // No declusterings passed soft drop in secondary plane
                lund_coords_events_secondary_x_sd.push_back({});
                lund_coords_events_secondary_y_sd.push_back({});
                lund_kt_events_secondary_sd.push_back({});
                lund_z_events_secondary_sd.push_back({});
                lund_psi_events_secondary_sd.push_back({});
                lund_delta_events_secondary_sd.push_back({});
                lund_mass_events_secondary_sd.push_back({});
                lund_psi12_events_sd.push_back(numeric_limits<double>::quiet_NaN());
                lund_max_kt_secondary_events_sd.push_back(-1);
                continue;
            }

            //double psi2 = declusts_secondary[max_declust_index].psi();
            fastjet::PseudoJet p3 = declusts_secondary[max_declust_index].harder();
            fastjet::PseudoJet p4 = declusts_secondary[max_declust_index].softer();
            lund_max_kt_secondary_events_sd.push_back(max_kt_index);

            double dpsi12 = cms_delta_phi(p1, p2, p3, p4);
            lund_psi12_events_sd.push_back(dpsi12);

            lund_coords_events_secondary_x_sd.push_back(lund_coords_jet_x);
            lund_coords_events_secondary_y_sd.push_back(lund_coords_jet_y);
            lund_kt_events_secondary_sd.push_back(lund_kt_jet);
            lund_z_events_secondary_sd.push_back(lund_z_jet);
            lund_psi_events_secondary_sd.push_back(lund_psi_jet);
            lund_delta_events_secondary_sd.push_back(lund_delta_jet);
            lund_mass_events_secondary_sd.push_back(lund_mass_jet);
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
        lund_branch_delta_sd->Fill();
        lund_branch_mass_sd->Fill();

        lund_branch_secondary_x_sd->Fill();
        lund_branch_secondary_y_sd->Fill();
        lund_branch_secondary_kt_sd->Fill();
        lund_branch_secondary_z_sd->Fill();
        lund_branch_secondary_psi_sd->Fill();
        lund_branch_secondary_delta_sd->Fill();
        lund_branch_secondary_mass_sd->Fill();

        lund_branch_psi12_sd->Fill();
        lund_branch_max_kt_sd->Fill();
        lund_branch_max_kt_secondary_sd->Fill();
    }

    // Write the tree and close the file
    tree->Write("", TObject::kOverwrite);
    file->Write("", TObject::kOverwrite);
    file->Close();

    cout << "Rank " << rank << ": finished processing " << event_count << " events.\n";
}