/*
g++ -std=c++17 -O2 \
main/lund_data.cpp -o build/lund_data \
$(fastjet-config --cxxflags) $(root-config --cflags) \
$(fastjet-config --libs) -lfastjetplugins -lfastjettools -lfastjetcontribfragile \
$(root-config --libs)
*/

//FastJet includes
#include "fastjet/PseudoJet.hh"
#include "fastjet/EECambridgePlugin.hh"
#include "fastjet/contrib/LundGenerator.hh"
#include "fastjet/contrib/IFNPlugin.hh"

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
#include <cstdlib>
#include <numeric>

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

static inline double delta_R(double eta1, double phi1, double eta2, double phi2) {
    double dphi = wrap_pm_pi(phi1 - phi2);
    double deta = eta1 - eta2;
    return sqrt(deta*deta + dphi*dphi);
}

enum ChannelLabel {
    kDeclustFail = -1,   // declustering failed (same as MC)
    kNoTruth     = -2,   // DATA: valid splitting but no gen truth to label it
    kUnmatched   = 0,
    kQQbar       = 1,
    kQG          = 2,
    kGG          = 3,
    kRest        = 4
};

struct SubjetShapeVars {
    int n_all = 0;
    int n_charged = 0;
    double sigma = 0.0;
    double ptD = 0.0;
};

static SubjetShapeVars compute_subjet_shapes(const fastjet::PseudoJet& subjet) {
    SubjetShapeVars sv;

    // get all constituents recursively from the CA clustering tree
    vector<fastjet::PseudoJet> consts = subjet.constituents();

    double sum_pt    = 0.0;
    double sum_pt2   = 0.0;
    double sum_pt_dr = 0.0;

    double axis_eta = subjet.eta();
    double axis_phi = subjet.phi();

    for (const auto& c : consts) {
        double pt = c.pt();
        if (pt <= 0) continue;

        sv.n_all++;

        if (c.user_index() != 0) sv.n_charged++;

        double deta = c.eta() - axis_eta;
        double dphi = wrap_pm_pi(c.phi() - axis_phi);
        double dR   = sqrt(deta*deta + dphi*dphi);

        sum_pt    += pt;
        sum_pt2   += pt * pt;
        sum_pt_dr += pt * dR;
    }

    if (sum_pt > 0) {
        sv.sigma = sum_pt_dr / sum_pt;       // pT-weighted width
        sv.ptD    = sum_pt2   / sum_pt;       // pT dispersion
    }

    return sv;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        cerr << "Usage: ./lund_data input.root [rank]\n";
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
    TTreeReaderValue<vector<float>> jet_eta(reader, "jet_eta");
    TTreeReaderValue<vector<float>> jet_phi(reader, "jet_phi");
    TTreeReaderValue<vector<vector<float>>> const_pt(reader, "const_pt");
    TTreeReaderValue<vector<vector<float>>> const_eta(reader, "const_eta");
    TTreeReaderValue<vector<vector<float>>> const_phi(reader, "const_phi");
    TTreeReaderValue<vector<vector<float>>> const_mass(reader, "const_mass");
    TTreeReaderValue<vector<vector<int>>> const_charge(reader, "const_charge");

    // NOTE: real data has no gen_* branches, so we do NOT read them.

    Long64_t nevents = tree->GetEntries();
    cout << "Number of events: " << nevents << endl;

    const double sd_beta = 0;     // 0 => mMDT // -1 => "traditional" soft drop (agressive)
    const double sd_beta_secondary = 0; // beta for secondary plane declustering (if different from primary)
    const double sd_zcut = 0.1;     // typical 0.05–0.2
    const double sd_zcut_secondary = 0.1; // beta for secondary plane declustering (if different from primary)
    const double R0   = 0.8;     // usually = jet R
    const double R0_antikt = 0.8;
    const double R0_ca = 2;
    const double soft_min_pt = 0; // Minimum pT for softer branch to pass soft drop condition (CMS-style)

    int event_count = 0;

    //Create lund generator and decluster the jet
    fastjet::contrib::LundGenerator lund;
    fastjet::JetDefinition jet_def(fastjet::cambridge_aachen_algorithm, R0_ca);

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
    vector< double> lund_max_kt_pt2_events_sd; //pT of softer branch in max kT declustering in primary plane

    vector< int> lund_primary_idx_sd; //Index of primary declustering in jet (DATA: sentinel only)
    vector< int> lund_secondary_idx_sd; //Index of secondary declustering in jet (DATA: sentinel only)

    //For the DNN
    vector<double> lund_p3_n_charged; //Number of charged particles in p3
    vector<double> lund_p4_n_charged; //Number of charged particles in p4
    vector<double> lund_p3_n_all; //Number of all particles in p3
    vector<double> lund_p4_n_all; //Number of all particles in p4
    vector<double> lund_p3_sigma; //Sigma of pT distribution of particles in p3
    vector<double> lund_p4_sigma; //Sigma of pT distribution of particles in p4
    vector<double> lund_p3_ptD; //PtD of particles in p3
    vector<double> lund_p4_ptD; //PtD of particles in p4

    //For the DNN
    vector<double> lund_p1_n_charged; //Number of charged particles in p1
    vector<double> lund_p2_n_charged; //Number of charged particles in p2
    vector<double> lund_p1_n_all; //Number of all particles in p1
    vector<double> lund_p2_n_all; //Number of all particles in p2
    vector<double> lund_p1_sigma; //Sigma of pT distribution of particles in p1
    vector<double> lund_p2_sigma; //Sigma of pT distribution of particles in p2
    vector<double> lund_p1_ptD; //PtD of particles in p1
    vector<double> lund_p2_ptD; //PtD of particles in p2

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

    auto lund_branch_primary_idx_sd = tree->Branch("lund_primary_idx_sd", &lund_primary_idx_sd);
    auto lund_branch_secondary_idx_sd = tree->Branch("lund_secondary_idx_sd", &lund_secondary_idx_sd);
    auto lund_branch_max_kt_pt2_sd = tree->Branch("lund_max_kt_pt2_sd", &lund_max_kt_pt2_events_sd);

    auto lund_branch_p3_n_charged = tree->Branch("lund_p3_n_charged", &lund_p3_n_charged);
    auto lund_branch_p4_n_charged = tree->Branch("lund_p4_n_charged", &lund_p4_n_charged);
    auto lund_branch_p3_n_all = tree->Branch("lund_p3_n_all", &lund_p3_n_all);
    auto lund_branch_p4_n_all = tree->Branch("lund_p4_n_all", &lund_p4_n_all);
    auto lund_branch_p3_sigma = tree->Branch("lund_p3_sigma", &lund_p3_sigma);
    auto lund_branch_p4_sigma = tree->Branch("lund_p4_sigma", &lund_p4_sigma);
    auto lund_branch_p3_ptD = tree->Branch("lund_p3_ptD", &lund_p3_ptD);
    auto lund_branch_p4_ptD = tree->Branch("lund_p4_ptD", &lund_p4_ptD);

    auto lund_branch_p1_n_charged = tree->Branch("lund_p1_n_charged", &lund_p1_n_charged);
    auto lund_branch_p2_n_charged = tree->Branch("lund_p2_n_charged", &lund_p2_n_charged);
    auto lund_branch_p1_n_all = tree->Branch("lund_p1_n_all", &lund_p1_n_all);
    auto lund_branch_p2_n_all = tree->Branch("lund_p2_n_all", &lund_p2_n_all);
    auto lund_branch_p1_sigma = tree->Branch("lund_p1_sigma", &lund_p1_sigma);
    auto lund_branch_p2_sigma = tree->Branch("lund_p2_sigma", &lund_p2_sigma);
    auto lund_branch_p1_ptD = tree->Branch("lund_p1_ptD", &lund_p1_ptD);
    auto lund_branch_p2_ptD = tree->Branch("lund_p2_ptD", &lund_p2_ptD);

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

        lund_primary_idx_sd.clear();
        lund_secondary_idx_sd.clear();
        lund_max_kt_pt2_events_sd.clear();

        lund_p3_n_charged.clear();
        lund_p4_n_charged.clear();
        lund_p3_n_all.clear();
        lund_p4_n_all.clear();
        lund_p3_sigma.clear();
        lund_p4_sigma.clear();
        lund_p3_ptD.clear();
        lund_p4_ptD.clear();

        lund_p1_n_charged.clear();
        lund_p2_n_charged.clear();
        lund_p1_n_all.clear();
        lund_p2_n_all.clear();
        lund_p1_sigma.clear();
        lund_p2_sigma.clear();
        lund_p1_ptD.clear();
        lund_p2_ptD.clear();

        // ================================================================
        // NO gen-level clustering in data (no gen particles, no truth).
        // ================================================================

        // Sort jet indices by descending pt (do not reorder branch vectors)
        vector<size_t> jet_indices(jet_pt->size());
        iota(jet_indices.begin(), jet_indices.end(), 0);
        sort(jet_indices.begin(), jet_indices.end(),[&](size_t i, size_t j) { return jet_pt->at(i) > jet_pt->at(j); });

        // max_kt indexes
        double max_kt;
        double max_kt_secondary;
        int max_kt_idx;
        int max_kt_secondary_idx;
        int idx_primary;
        int idx_secondary;
        int passes_primary;
        int passes_secondary;
        int passes_primary_store = 0;
        int passes_secondary_store = 0;

        for (size_t ord=0; ord < jet_indices.size(); ++ord){
            const size_t ijet = jet_indices[ord]; // Get the original index of the jet in the branch vectors

            vector<fastjet::PseudoJet> constituents;
            constituents.reserve(const_pt->at(ijet).size());

            for (size_t iconst=0; iconst < const_pt->at(ijet).size(); ++iconst){
                double pt = const_pt->at(ijet)[iconst];
                double eta = const_eta->at(ijet)[iconst];
                double phi = const_phi->at(ijet)[iconst];
                double mass = const_mass->at(ijet)[iconst];

                double px = pt * cos(phi);
                double py = pt * sin(phi);
                double pz = pt * sinh(eta);
                double E  = sqrt(px*px + py*py + pz*pz + mass*mass);

                fastjet::PseudoJet p(px, py, pz, E);
                p.set_user_index(const_charge->at(ijet)[iconst]); // Store charge in user index
                constituents.push_back(p);
            }

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

                lund_primary_idx_sd.push_back(kDeclustFail);
                lund_secondary_idx_sd.push_back(kDeclustFail);
                lund_max_kt_pt2_events_sd.push_back(numeric_limits<double>::quiet_NaN());

                lund_p3_n_charged.push_back(-1);
                lund_p4_n_charged.push_back(-1);
                lund_p3_n_all.push_back(-1);
                lund_p4_n_all.push_back(-1);
                lund_p3_sigma.push_back(numeric_limits<double>::quiet_NaN());
                lund_p4_sigma.push_back(numeric_limits<double>::quiet_NaN());
                lund_p3_ptD.push_back(numeric_limits<double>::quiet_NaN());
                lund_p4_ptD.push_back(numeric_limits<double>::quiet_NaN());

                lund_p1_n_charged.push_back(-1);
                lund_p2_n_charged.push_back(-1);
                lund_p1_n_all.push_back(-1);
                lund_p2_n_all.push_back(-1);
                lund_p1_sigma.push_back(numeric_limits<double>::quiet_NaN());
                lund_p2_sigma.push_back(numeric_limits<double>::quiet_NaN());
                lund_p1_ptD.push_back(numeric_limits<double>::quiet_NaN());
                lund_p2_ptD.push_back(numeric_limits<double>::quiet_NaN());
                continue;
            }

            fastjet::ClusterSequence cs(constituents, jet_def);
            auto jets = fastjet::sorted_by_pt(cs.inclusive_jets());
            if (jets.size() < 1){
                continue; // No jets found, skip to next jet
            }
            if (jets.size() > 1) {
                cerr << "Warning: more than one jet found in clustering. This should not happen with the CA reclustering. Skipping this jet.\n";
            }

            // Decluster the leading jet using the Lund generator
            lund_coords_jet_x.clear();
            lund_coords_jet_y.clear();
            lund_kt_jet.clear();
            lund_z_jet.clear();
            lund_psi_jet.clear();
            lund_delta_jet.clear();
            lund_mass_jet.clear();

            vector<fastjet::contrib::LundDeclustering> declusters = lund.result(jets[0]);

            max_kt = -1;
            max_kt_idx = -1;
            idx_primary = 0;
            passes_primary = 0;
            for (const auto& decl : declusters) {
                double z = decl.z();
                double delta = decl.Delta();
                fastjet::PseudoJet soft = decl.softer();
                double soft_pt = soft.pt();
                bool passes = (z > sd_zcut * pow(delta/R0, sd_beta)) && (soft_pt > soft_min_pt);

                if (passes){
                    pair<double,double> coords = decl.lund_coordinates();
                    double kt = decl.kt();
                    double psi = decl.psi();
                    double mass = decl.m();

                    if (kt > max_kt) {
                        max_kt = kt;
                        max_kt_idx = idx_primary;
                        passes_primary_store = passes_primary;
                    }

                    lund_coords_jet_x.push_back(coords.first);
                    lund_coords_jet_y.push_back(coords.second);
                    lund_kt_jet.push_back(kt);
                    lund_z_jet.push_back(z);
                    lund_psi_jet.push_back(psi);
                    lund_delta_jet.push_back(delta);
                    lund_mass_jet.push_back(mass);
                    passes_primary++;
                }
                idx_primary++;
            }

            if (max_kt < 0) {
                // No declusterings passed soft drop, fill with empty values
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

                lund_primary_idx_sd.push_back(kDeclustFail);
                lund_secondary_idx_sd.push_back(kDeclustFail);
                lund_max_kt_pt2_events_sd.push_back(numeric_limits<double>::quiet_NaN());

                lund_p3_n_charged.push_back(-1);
                lund_p4_n_charged.push_back(-1);
                lund_p3_n_all.push_back(-1);
                lund_p4_n_all.push_back(-1);
                lund_p3_sigma.push_back(numeric_limits<double>::quiet_NaN());
                lund_p4_sigma.push_back(numeric_limits<double>::quiet_NaN());
                lund_p3_ptD.push_back(numeric_limits<double>::quiet_NaN());
                lund_p4_ptD.push_back(numeric_limits<double>::quiet_NaN());

                lund_p1_n_charged.push_back(-1);
                lund_p2_n_charged.push_back(-1);
                lund_p1_n_all.push_back(-1);
                lund_p2_n_all.push_back(-1);
                lund_p1_sigma.push_back(numeric_limits<double>::quiet_NaN());
                lund_p2_sigma.push_back(numeric_limits<double>::quiet_NaN());
                lund_p1_ptD.push_back(numeric_limits<double>::quiet_NaN());
                lund_p2_ptD.push_back(numeric_limits<double>::quiet_NaN());
                continue;
            }

            lund_coords_events_x_sd.push_back(lund_coords_jet_x);
            lund_coords_events_y_sd.push_back(lund_coords_jet_y);
            lund_kt_events_sd.push_back(lund_kt_jet);
            lund_z_events_sd.push_back(lund_z_jet);
            lund_psi_events_sd.push_back(lund_psi_jet);
            lund_delta_events_sd.push_back(lund_delta_jet);
            lund_mass_events_sd.push_back(lund_mass_jet);
            lund_max_kt_events_sd.push_back(passes_primary_store);

            lund_coords_jet_x.clear();
            lund_coords_jet_y.clear();
            lund_kt_jet.clear();
            lund_z_jet.clear();
            lund_psi_jet.clear();
            lund_delta_jet.clear();
            lund_mass_jet.clear();

            fastjet::PseudoJet p1 = declusters[max_kt_idx].harder();
            fastjet::PseudoJet p2 = declusters[max_kt_idx].softer();

            lund_max_kt_pt2_events_sd.push_back(p2.pt());

            vector<fastjet::contrib::LundDeclustering> declusters_secondary = lund.result(p2);

            max_kt_secondary = -1;
            max_kt_secondary_idx = -1;
            idx_secondary = 0;
            passes_secondary = 0;
            for (const auto& decl : declusters_secondary) {
                double z = decl.z();
                double delta = decl.Delta();
                bool passes = (z > sd_zcut_secondary * pow(delta/R0, sd_beta_secondary));

                if (passes){
                    pair<double,double> coords = decl.lund_coordinates();
                    double kt = decl.kt();
                    double psi = decl.psi();
                    double mass = decl.m();

                    if (kt > max_kt_secondary) {
                        max_kt_secondary = kt;
                        max_kt_secondary_idx = idx_secondary;
                        passes_secondary_store = passes_secondary;
                    }

                    lund_coords_jet_x.push_back(coords.first);
                    lund_coords_jet_y.push_back(coords.second);
                    lund_kt_jet.push_back(kt);
                    lund_z_jet.push_back(z);
                    lund_psi_jet.push_back(psi);
                    lund_delta_jet.push_back(delta);
                    lund_mass_jet.push_back(mass);
                    passes_secondary++;
                }
                idx_secondary++;
            }

            if (max_kt_secondary < 0) {
                // No declusterings passed secondary soft drop, fill with empty values
                lund_coords_events_secondary_x_sd.push_back({});
                lund_coords_events_secondary_y_sd.push_back({});
                lund_kt_events_secondary_sd.push_back({});
                lund_z_events_secondary_sd.push_back({});
                lund_psi_events_secondary_sd.push_back({});
                lund_delta_events_secondary_sd.push_back({});
                lund_mass_events_secondary_sd.push_back({});

                lund_psi12_events_sd.push_back(numeric_limits<double>::quiet_NaN());
                lund_max_kt_secondary_events_sd.push_back(-1);

                lund_primary_idx_sd.push_back(kDeclustFail);
                lund_secondary_idx_sd.push_back(kDeclustFail);

                lund_p3_n_charged.push_back(-1);
                lund_p4_n_charged.push_back(-1);
                lund_p3_n_all.push_back(-1);
                lund_p4_n_all.push_back(-1);
                lund_p3_sigma.push_back(numeric_limits<double>::quiet_NaN());
                lund_p4_sigma.push_back(numeric_limits<double>::quiet_NaN());
                lund_p3_ptD.push_back(numeric_limits<double>::quiet_NaN());
                lund_p4_ptD.push_back(numeric_limits<double>::quiet_NaN());

                lund_p1_n_charged.push_back(-1);
                lund_p2_n_charged.push_back(-1);
                lund_p1_n_all.push_back(-1);
                lund_p2_n_all.push_back(-1);
                lund_p1_sigma.push_back(numeric_limits<double>::quiet_NaN());
                lund_p2_sigma.push_back(numeric_limits<double>::quiet_NaN());
                lund_p1_ptD.push_back(numeric_limits<double>::quiet_NaN());
                lund_p2_ptD.push_back(numeric_limits<double>::quiet_NaN());
                continue;
            }

            lund_coords_events_secondary_x_sd.push_back(lund_coords_jet_x);
            lund_coords_events_secondary_y_sd.push_back(lund_coords_jet_y);
            lund_kt_events_secondary_sd.push_back(lund_kt_jet);
            lund_z_events_secondary_sd.push_back(lund_z_jet);
            lund_psi_events_secondary_sd.push_back(lund_psi_jet);
            lund_delta_events_secondary_sd.push_back(lund_delta_jet);
            lund_mass_events_secondary_sd.push_back(lund_mass_jet);

            fastjet::PseudoJet p3 = declusters_secondary[max_kt_secondary_idx].harder();
            fastjet::PseudoJet p4 = declusters_secondary[max_kt_secondary_idx].softer();

            double psi12 = cms_delta_phi(p1, p2, p3, p4);
            lund_psi12_events_sd.push_back(psi12);
            lund_max_kt_secondary_events_sd.push_back(passes_secondary_store);

            // ================================================================
            // DATA: no gen truth. A valid primary+secondary splitting exists,
            // so we mark the channel labels with kNoTruth (-2) rather than a
            // real channel. All reco DNN variables below ARE filled normally.
            // ================================================================
            lund_primary_idx_sd.push_back(kNoTruth);
            lund_secondary_idx_sd.push_back(kNoTruth);

            SubjetShapeVars sv_p3 = compute_subjet_shapes(p3);
            SubjetShapeVars sv_p4 = compute_subjet_shapes(p4);

            lund_p3_n_charged.push_back(sv_p3.n_charged);
            lund_p4_n_charged.push_back(sv_p4.n_charged);
            lund_p3_n_all.push_back(sv_p3.n_all);
            lund_p4_n_all.push_back(sv_p4.n_all);
            lund_p3_sigma.push_back(sv_p3.sigma);
            lund_p4_sigma.push_back(sv_p4.sigma);
            lund_p3_ptD.push_back(sv_p3.ptD);
            lund_p4_ptD.push_back(sv_p4.ptD);

            SubjetShapeVars sv_p1 = compute_subjet_shapes(p1);
            SubjetShapeVars sv_p2 = compute_subjet_shapes(p2);

            lund_p1_n_charged.push_back(sv_p1.n_charged);
            lund_p2_n_charged.push_back(sv_p2.n_charged);
            lund_p1_n_all.push_back(sv_p1.n_all);
            lund_p2_n_all.push_back(sv_p2.n_all);
            lund_p1_sigma.push_back(sv_p1.sigma);
            lund_p2_sigma.push_back(sv_p2.sigma);
            lund_p1_ptD.push_back(sv_p1.ptD);
            lund_p2_ptD.push_back(sv_p2.ptD);
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

        lund_branch_primary_idx_sd->Fill();
        lund_branch_secondary_idx_sd->Fill();
        lund_branch_max_kt_pt2_sd->Fill();

        lund_branch_p3_n_charged->Fill();
        lund_branch_p4_n_charged->Fill();
        lund_branch_p3_n_all->Fill();
        lund_branch_p4_n_all->Fill();
        lund_branch_p3_sigma->Fill();
        lund_branch_p4_sigma->Fill();
        lund_branch_p3_ptD->Fill();
        lund_branch_p4_ptD->Fill();

        lund_branch_p1_n_charged->Fill();
        lund_branch_p2_n_charged->Fill();
        lund_branch_p1_n_all->Fill();
        lund_branch_p2_n_all->Fill();
        lund_branch_p1_sigma->Fill();
        lund_branch_p2_sigma->Fill();
        lund_branch_p1_ptD->Fill();
        lund_branch_p2_ptD->Fill();
    }

    // Write the tree and close the file
    tree->Write("", TObject::kOverwrite);
    file->Write("", TObject::kOverwrite);
    file->Close();

    cout << "Rank " << rank << ": finished processing " << event_count << " events.\n";
}