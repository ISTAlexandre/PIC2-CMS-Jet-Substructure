/*
g++ -std=c++17 -O2 \
main/lund_gen.cpp -o build/lund_gen \
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

static inline int channel_from_pdgid(int pdg_id1, int pdg_id2) {
    // Simple categorization of partonic channels based on PDG IDs)
    bool is_quark1 = (1 <= abs(pdg_id1)) && (abs(pdg_id1) <= 6); // u, d, s, c, b
    bool is_quark2 = (1 <= abs(pdg_id2)) && (abs(pdg_id2) <= 6);
    bool is_gluon1 = pdg_id1 == 21;
    bool is_gluon2 = pdg_id2 == 21;

    if (is_quark1 && is_quark2) return 1; // quark-quark
    else if ((is_quark1 && is_gluon2) || (is_gluon1 && is_quark2)) return 2; // quark-gluon
    else if (is_gluon1 && is_gluon2) return 3; // gluon-gluon
    else if (pdg_id1 == 0 and pdg_id2 == 0) return 4; // unknown/other

    else if ((pdg_id1 == 0 && is_quark2) || (is_quark1 && pdg_id2 == 0)) return 2; // one unknown/other and one quark
    else if ((pdg_id1 == 0 && is_gluon2) || (is_gluon1 && pdg_id2 == 0)) return 3; // one unknown/other and one gluon
    else return 4; // fallback to unknown/other for any other cases

}

static inline double delta_R(double eta1, double phi1, double eta2, double phi2) {
    double dphi = wrap_pm_pi(phi1 - phi2);
    double deta = eta1 - eta2;
    return sqrt(deta*deta + dphi*dphi);
}

enum ChannelLabel {
    kDeclustFail = -1,
    kUnmatched   = 0,
    kQQbar       = 1,
    kQG          = 2,
    kGG          = 3,
    kRest        = 4
};

struct SingleFlav {
    bool ok = false;
    int iflav = 0;   // 1=d, 2=u, 3=s, 4=c, 5=b
    int sign = 0;    // +1 = q, -1 = qbar
};

static inline SingleFlav get_single_flavour(const fastjet::contrib::FlavInfo& f) {
    int n_nonzero = 0;
    int iflav_nonzero = 0;
    int sign_nonzero = 0;
    int abs_sum = 0;

    for (int iflav = 1; iflav <= 6; ++iflav) {   // exclude top
        int v = f[iflav];
        if (v != 0) {
            ++n_nonzero;
            iflav_nonzero = iflav;
            sign_nonzero = (v > 0) ? +1 : -1;
            abs_sum += std::abs(v);
        }
    }

    // exactly one unit of exactly one flavour -> quark-like singlet
    if (n_nonzero == 1 && abs_sum == 1) {
        return {true, iflav_nonzero, sign_nonzero};
    }

    return {};
}

static inline int channel_from_flavinfo(const fastjet::contrib::FlavInfo& f1,
                                        const fastjet::contrib::FlavInfo& f2) {
    const bool g1 = f1.is_flavourless();
    const bool g2 = f2.is_flavourless();

    const SingleFlav q1 = get_single_flavour(f1);
    const SingleFlav q2 = get_single_flavour(f2);

    // gg
    if (g1 && g2) return kGG;

    // qg
    if ((q1.ok && g2) || (g1 && q2.ok)) return kQG;

    // q qbar only if same flavour, opposite sign
    if (q1.ok && q2.ok) {
        if (q1.iflav == q2.iflav && q1.sign == -q2.sign) {
            return kQQbar;
        }
        return kRest;   // qq, q q', qbar qbar, etc.
    }

    // anything multi-flavoured / non-singlet / ambiguous
    return kRest;
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

    TTreeReaderValue<vector<float>> gen_pt(reader, "gen_pt");
    TTreeReaderValue<vector<float>> gen_eta(reader, "gen_eta");
    TTreeReaderValue<vector<float>> gen_phi(reader, "gen_phi");
    TTreeReaderValue<vector<float>> gen_mass(reader, "gen_mass");
    TTreeReaderValue<vector<int>> gen_pdgId(reader, "gen_pdgId");

    Long64_t nevents = tree->GetEntries();
    cout << "Number of events: " << nevents << endl;

    const double sd_beta = 0;     // 0 => mMDT // -1 => "traditional" soft drop (agressive)
    const double sd_beta_secondary = 0; // beta for secondary plane declustering (if different from primary)
    const double sd_zcut = 0;     // typical 0.05–0.2
    const double sd_zcut_secondary = 0.1; // beta for secondary plane declustering (if different from primary)
    const double R0   = 0.8;     // usually = jet R
    const double soft_min_pt = 130; // Minimum pT for softer branch to pass soft drop condition (CMS-style)

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
    vector< double> lund_max_kt_pt2_events_sd; //pT of softer branch in max kT declustering in primary plane

    vector< int> lund_primary_idx_sd; //Index of primary declustering in jet
    vector< int> lund_secondary_idx_sd; //Index of secondary declustering in jet

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

        vector<fastjet::PseudoJet> gen_particles;
        for (size_t igenpt=0; igenpt < gen_pt->size(); ++igenpt) {
            double gen_pt_val = gen_pt->at(igenpt);
            double gen_eta_val = gen_eta->at(igenpt);
            double gen_phi_val = gen_phi->at(igenpt);
            double gen_mass_val = gen_mass->at(igenpt);

            double gen_px = gen_pt_val * cos(gen_phi_val);
            double gen_py = gen_pt_val * sin(gen_phi_val);
            double gen_pz = gen_pt_val * sinh(gen_eta_val);
            double gen_E  = sqrt(gen_px*gen_px + gen_py*gen_py + gen_pz*gen_pz + gen_mass_val*gen_mass_val);

            fastjet::PseudoJet gen_p(gen_px, gen_py, gen_pz, gen_E);
            int pdg_id = gen_pdgId->at(igenpt);
            gen_p.set_user_info(new fastjet::contrib::FlavInfo(pdg_id)); // Store PDG ID in user index for later use in channel classification
            gen_particles.push_back(gen_p);
        }

        //Create lund generator and decluster the jet
        fastjet::contrib::LundGenerator lund;
        fastjet::JetDefinition jet_def(fastjet::cambridge_aachen_algorithm, R0);

        //Gen-level declustering for primary and secondary planes to identify partonic channels
        double alpha = 2.0;
        fastjet::JetDefinition gen_jet_def(new fastjet::contrib::IFNPlugin(jet_def, alpha));
        gen_jet_def.delete_plugin_when_unused();
        fastjet::ClusterSequence gen_cs(gen_particles, gen_jet_def);
        auto gen_jets = fastjet::sorted_by_pt(gen_cs.inclusive_jets());

        vector<bool> used_gen_jets(gen_jets.size(), false);

        // max_kt indexes
        double max_kt;
        double max_kt_secondary;
        int max_kt_idx;
        int max_kt_secondary_idx;
        int idx_primary;
        int idx_secondary;
        int passes_primary;
        int passes_secondary;
        int passes_primary_store;
        int passes_secondary_store;
        
        // Sort jet indices by descending pt (do not reorder branch vectors)
        vector<size_t> jet_indices(jet_pt->size());
        iota(jet_indices.begin(), jet_indices.end(), 0);
        sort(jet_indices.begin(), jet_indices.end(),[&](size_t i, size_t j) { return jet_pt->at(i) > jet_pt->at(j); });
        
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

                lund_primary_idx_sd.push_back(-1);
                lund_secondary_idx_sd.push_back(-1);
                lund_max_kt_pt2_events_sd.push_back(numeric_limits<double>::quiet_NaN());
                continue;
            }

            fastjet::ClusterSequence cs(constituents, jet_def);
            auto jets = fastjet::sorted_by_pt(cs.inclusive_jets());
            fastjet::PseudoJet leading_jet = jets[0];
            
            double minor_dR = R0/2;
            int best_gen_idx = -1;

            for (size_t igenjet=0; igenjet < gen_jets.size(); ++igenjet) {
                if (used_gen_jets[igenjet]) continue; // Skip already matched gen jets

                double dR = delta_R(leading_jet.eta(), leading_jet.phi(), gen_jets[igenjet].eta(), gen_jets[igenjet].phi());
                if (dR < minor_dR) {
                    minor_dR = dR;
                    best_gen_idx = igenjet;
                }
            }

            fastjet::PseudoJet gen_jet;
            if (best_gen_idx >= 0){
                gen_jet = gen_jets[best_gen_idx];
                used_gen_jets[best_gen_idx] = true; // Mark this gen jet as used
            }

            // Decluster the leading jet using the Lund generator
            lund_coords_jet_x.clear();
            lund_coords_jet_y.clear();
            lund_kt_jet.clear();
            lund_z_jet.clear();
            lund_psi_jet.clear();
            lund_delta_jet.clear();
            lund_mass_jet.clear();

            vector<fastjet::contrib::LundDeclustering> declusters = lund.result(leading_jet);

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

                lund_primary_idx_sd.push_back(-1);
                lund_secondary_idx_sd.push_back(-1);
                lund_max_kt_pt2_events_sd.push_back(numeric_limits<double>::quiet_NaN());
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

                lund_primary_idx_sd.push_back(-1);
                lund_secondary_idx_sd.push_back(-1);

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

            // Decluster gen_jet for channel identification

            if (best_gen_idx < 0) {
                // No matched gen jet, fill with empty values and continue
                //cout << "No matched gen jet for event " << event_count << ", skipping declustering-based channel ID\n";
                lund_primary_idx_sd.push_back(-1);
                lund_secondary_idx_sd.push_back(-1);
                continue;
            }

            fastjet::PseudoJet a;
            fastjet::PseudoJet b;
            fastjet::PseudoJet gen_p1;
            fastjet::PseudoJet gen_p2;
            max_kt = -1;
            while (gen_jet.has_parents(a, b)) {
                double z = min(a.pt(), b.pt()) / (a.pt() + b.pt());
                double delta = delta_R(a.eta(), a.phi(), b.eta(), b.phi());
                bool passes = (z > sd_zcut * pow(delta/R0, sd_beta)) && (min(a.pt(), b.pt()) > soft_min_pt);
                fastjet::PseudoJet harder = (a.pt() > b.pt()) ? a : b;
                fastjet::PseudoJet softer = (a.pt() > b.pt()) ? b : a;
                if (passes){
                    if (max_kt < softer.pt() * delta) {
                        max_kt = softer.pt() * delta;
                        gen_p1 = harder;
                        gen_p2 = softer;
                    }
                }
                gen_jet = harder;
            }

            //Continue if gen-level declustering did not find a valid splitting
            if (gen_p1.pt() == 0 || gen_p2.pt() == 0) {
                lund_primary_idx_sd.push_back(-1);
                lund_secondary_idx_sd.push_back(-1);
                continue;
            }

            auto f1 = fastjet::contrib::FlavHistory::current_flavour_of(gen_p1);
            auto f2 = fastjet::contrib::FlavHistory::current_flavour_of(gen_p2);
            int channel1 = channel_from_flavinfo(f1,f2);
            
            double reco_delta = delta_R(p1.eta(), p1.phi(), p2.eta(), p2.phi());
            double d11 = delta_R(p1.eta(), p1.phi(), gen_p1.eta(), gen_p1.phi());
            double d12 = delta_R(p1.eta(), p1.phi(), gen_p2.eta(), gen_p2.phi());
            double d21 = delta_R(p2.eta(), p2.phi(), gen_p1.eta(), gen_p1.phi());
            double d22 = delta_R(p2.eta(), p2.phi(), gen_p2.eta(), gen_p2.phi());
            
            bool dir = d11 < 0.5 * reco_delta && d22 < 0.5 * reco_delta;
            bool swap = d12 < 0.5 * reco_delta && d21 < 0.5 * reco_delta;

            if (dir || swap) {
                lund_primary_idx_sd.push_back(channel1);
            } else {
                lund_primary_idx_sd.push_back(0); // Unmatched
            }
            
            // Decluster gen_jet for channel identification
            fastjet::PseudoJet gen_p3;
            fastjet::PseudoJet gen_p4;
            max_kt_secondary = -1;
            while (gen_p2.has_parents(a, b)) {
                double z = min(a.pt(), b.pt()) / (a.pt() + b.pt());
                double delta = delta_R(a.eta(), a.phi(), b.eta(), b.phi());
                bool passes = (z > sd_zcut_secondary * pow(delta/R0, sd_beta_secondary));
                fastjet::PseudoJet harder = (a.pt() > b.pt()) ? a : b;
                fastjet::PseudoJet softer = (a.pt() > b.pt()) ? b : a;
                if (passes){
                    if (max_kt_secondary < softer.pt() * delta) {
                        max_kt_secondary = softer.pt() * delta;
                        gen_p3 = harder;
                        gen_p4 = softer;
                    }
                }
                gen_p2 = harder;
                
            }
            
            //Continue if gen-level declustering did not find a valid splitting
            if (gen_p3.pt() == 0 || gen_p4.pt() == 0) {
                lund_secondary_idx_sd.push_back(-1);
                continue;
            }

            auto f3 = fastjet::contrib::FlavHistory::current_flavour_of(gen_p3);
            auto f4 = fastjet::contrib::FlavHistory::current_flavour_of(gen_p4);
            int channel2 = channel_from_flavinfo(f3,f4);
            double reco_delta_secondary = delta_R(p3.eta(), p3.phi(), p4.eta(), p4.phi());
            
            double d33 = delta_R(p3.eta(), p3.phi(), gen_p3.eta(), gen_p3.phi());
            double d34 = delta_R(p3.eta(), p3.phi(), gen_p4.eta(), gen_p4.phi());
            double d43 = delta_R(p4.eta(), p4.phi(), gen_p3.eta(), gen_p3.phi());
            double d44 = delta_R(p4.eta(), p4.phi(), gen_p4.eta(), gen_p4.phi());

            bool direct_match = d33 < 0.5 *reco_delta_secondary && d44 < 0.5 * reco_delta_secondary;
            bool swapped_match = d34 < 0.5 * reco_delta_secondary && d43 < 0.5 * reco_delta_secondary;
            if (direct_match || swapped_match) {
                lund_secondary_idx_sd.push_back(channel2);
            } else {
                lund_secondary_idx_sd.push_back(0); // Unmatched
            }

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
    }

    // Write the tree and close the file
    tree->Write("", TObject::kOverwrite);
    file->Write("", TObject::kOverwrite);
    file->Close();

    cout << "Rank " << rank << ": finished processing " << event_count << " events.\n";
}