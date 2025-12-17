/*
g++ -std=c++17 -O2 \
  main/lund_plane.cpp -o build/lund_plane \
  $(fastjet-config --cxxflags) $(root-config --cflags) \
  $(fastjet-config --libs) -lfastjettools -lfastjetcontribfragile \
  $(root-config --libs)
*/

#include <fastjet/ClusterSequence.hh>
#include <fastjet/PseudoJet.hh>
#include <iostream>
#include <vector>
#include <fastjet/contrib/LundGenerator.hh>
#include <fastjet/contrib/LundPlane.hh>
#include "fastjet/contrib/LundWithSecondary.hh"

#include "TFile.h"
#include "TTree.h"
#include "TKey.h"
#include <TTreeReader.h>
#include <TTreeReaderValue.h>
#include <fastjet/contrib/LundJSON.hh>
#include "fastjet/contrib/SoftDrop.hh"

#include <cmath>
#include <TH1D.h>
#include <TH1F.h>
#include <TCanvas.h>

struct SplitVars {
    double lambda_val;
    double kt;
    double mass;
    double z;
    double kappa;
    double psi;
};

inline double deltaR(const fastjet::PseudoJet& a, const fastjet::PseudoJet& b) {
    // uses rap() and wrapped Δφ internally
    return a.delta_R(b);
}

inline double y(const fastjet::PseudoJet& p) {
    const double num = p.E() + p.pz();
    const double den = p.E() - p.pz();
    return std::log(num / den);
}

inline double lambda_ab(const fastjet::PseudoJet& a, const fastjet::PseudoJet& b) {
    return deltaR(a, b);
}

inline double z(const fastjet::PseudoJet& a, const fastjet::PseudoJet& b) {
    const double pa = a.pt(), pb = b.pt();
    const double sum = pa + pb;
    return sum > 0.0 ? std::min(pa, pb) / sum : 0.0;
}

inline double mass_calculator(const fastjet::PseudoJet& p) {
    const double m2 = p.E() * p.E() - p.px() * p.px() - p.py() * p.py() - p.pz() * p.pz();
    return std::sqrt(std::max(0.0, m2));
}

inline double m(const fastjet::PseudoJet& a, const fastjet::PseudoJet& b) {
    const double ma2 = std::pow(mass_calculator(a), 2);
    const double mb2 = std::pow(mass_calculator(b), 2);
    const double cross = a.E() * b.E() - a.px() * b.px() - a.py() * b.py() - a.pz() * b.pz();
    const double m2 = ma2 + mb2 + 2.0 * cross;
    return std::sqrt(std::max(0.0, m2));
}

inline double psi(const fastjet::PseudoJet& a, const fastjet::PseudoJet& b) {
    const double dy   = y(b) - y(a);
    const double dphi = b.phi() - a.phi();
    // safer than atan(dy/dphi)
    return std::atan2(dy, dphi);
}

inline double k_t(const fastjet::PseudoJet& a, const fastjet::PseudoJet& b) {
    return std::min(a.pt(), b.pt()) * deltaR(a, b);
}

inline double kappa(const fastjet::PseudoJet& a, const fastjet::PseudoJet& b) {
    return z(a, b) * lambda_ab(a, b);
}

inline SplitVars dic_var(const fastjet::PseudoJet& a, const fastjet::PseudoJet& b) {
    SplitVars out;
    out.lambda_val = lambda_ab(a, b);
    out.kt         = k_t(a, b);
    out.mass       = m(a, b);
    out.z          = z(a, b);
    out.kappa      = kappa(a, b);
    out.psi        = psi(a, b);
    return out;
}

inline bool compare_jets(const fastjet::PseudoJet& j1, const fastjet::PseudoJet& j2) {
    return j1.px() == j2.px() && j1.py() == j2.py() && j1.pz() == j2.pz() && j1.E() == j2.E();
}

struct V3 { double x,y,z; };

inline V3 v3(const fastjet::PseudoJet& p){ return {p.px(), p.py(), p.pz()}; }

inline V3 cross(const V3& a, const V3& b){
  return { a.y*b.z - a.z*b.y,
           a.z*b.x - a.x*b.z,
           a.x*b.y - a.y*b.x };
}

inline double dot(const V3& a, const V3& b){ return a.x*b.x + a.y*b.y + a.z*b.z; }

inline double norm(const V3& a){ return std::sqrt(dot(a,a)); }

inline V3 unit(const V3& a){
  double n = norm(a);
  if (n <= 0) return {0,0,0};
  return {a.x/n, a.y/n, a.z/n};
}

inline double wrap_angle(double a){
  const double twoPi = 2.0*M_PI;
  a = std::fmod(a + M_PI, twoPi);
  if (a < 0) a += twoPi;
  return a - M_PI;
}

// signed Δψ between normals n_prev and n_cur, sign from (n_prev×n_cur)·p_hard
inline double signed_dpsi(const V3& n_prev, const V3& n_cur, const V3& p_hard){
  V3 c = cross(n_prev, n_cur);
  double ang = std::atan2(norm(c), dot(n_prev, n_cur)); // [0,pi]
  if (dot(c, p_hard) < 0.0) ang = -ang;
  return ang; // DO NOT wrap here
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: ./lund_plane input.root [rank]\n";
        return 1;
    }
    const char* path = argv[1];
    int rank = (argc > 2) ? std::atoi(argv[2]) : -1;
    std::cout << "Processing file " << path << " (passed rank=" << rank << ")\n";

    TFile* file = TFile::Open(path,"UPDATE");
    if (!file || file->IsZombie()) {
        std::cerr << "Error: could not open file " << path << std::endl;
        return 1;
    }
    TTree* tree = dynamic_cast<TTree*>(file->Get("jetTree"));
    if (!tree) {
        std::cerr << "Error: could not find TTree 'jetTree' in file " << path << std::endl;
        file->Close();
        return 1;
    }

    TTreeReader reader(tree);
    TTreeReaderValue<std::vector<float>> jet_pt(reader, "jet_pt");
    //TTreeReaderValue<std::vector<int>> jetAK(reader, "jetAK");
    TTreeReaderValue<std::vector<std::vector<float>>> const_pt(reader, "const_pt");
    TTreeReaderValue<std::vector<std::vector<float>>> const_eta(reader, "const_eta");
    TTreeReaderValue<std::vector<std::vector<float>>> const_phi(reader, "const_phi");
    TTreeReaderValue<std::vector<std::vector<float>>> const_mass(reader, "const_mass");

    //Create ROOT histogram
    TH1D delta12_hist = TH1D("delta12_hist", "Delta Psi between first and second hardest splittings;#Delta#psi_{12};Entries", 25, -M_PI, M_PI);

    //Create branch for lund plane coordinates
    vector< vector< double > > lund_coords_events_x;
    vector< vector< double > > lund_coords_events_y;
    vector< vector< double > > lund_delta_events;
    vector< vector< double > > lund_kt_events;
    vector< vector< double > > lund_z_events;
    vector< vector< double > > lund_psi_events;
    vector< vector< double > > lund_kappa_events;
    vector< vector< double > > lund_mass_events;
    //vector< vector< double > > lund_phi_events;

    vector< double > lund_coords_jet_x;
    vector< double > lund_coords_jet_y;
    vector< double > lund_delta_jet;
    vector< double > lund_kt_jet;
    vector< double > lund_z_jet;
    vector< double > lund_psi_jet;
    vector< double > lund_kappa_jet;
    vector< double > lund_mass_jet;
    //vector< double > lund_phi_jet;

    //Create branch for secondary lund plane coordinates
    vector< vector< double > > lund_coords_events_secondary_x;
    vector< vector< double > > lund_coords_events_secondary_y;
    vector< vector< double > > lund_delta_events_secondary;
    vector< vector< double > > lund_kt_events_secondary;
    vector< vector< double > > lund_z_events_secondary;
    vector< vector< double > > lund_psi_events_secondary;
    vector< vector< double > > lund_kappa_events_secondary;
    vector< vector< double > > lund_mass_events_secondary;
    //vector< vector< double > > lund_phi_events_secondary;

    vector< double > lund_coords_secondary_x;
    vector< double > lund_coords_secondary_y;
    vector< double > lund_delta_jet_secondary;
    vector< double > lund_kt_jet_secondary;
    vector< double > lund_z_jet_secondary;
    vector< double > lund_psi_jet_secondary;
    vector< double > lund_kappa_jet_secondary;
    vector< double > lund_mass_jet_secondary;
    //vector< double > lund_phi_jet_secondary;

    //Create branch for soft drop primary lund plane coordinates
    vector< vector< double > > lund_coords_events_x_sd;
    vector< vector< double > > lund_coords_events_y_sd;
    vector< vector< double > > lund_delta_events_sd;
    vector< vector< double > > lund_kt_events_sd;
    vector< vector< double > > lund_z_events_sd;
    vector< vector< double > > lund_psi_events_sd;
    vector< vector< double > > lund_kappa_events_sd;
    vector< vector< double > > lund_mass_events_sd;

    vector< double > lund_coords_jet_x_sd;
    vector< double > lund_coords_jet_y_sd;
    vector< double > lund_delta_jet_sd;
    vector< double > lund_kt_jet_sd;
    vector< double > lund_z_jet_sd;
    vector< double > lund_psi_jet_sd;
    vector< double > lund_kappa_jet_sd;
    vector< double > lund_mass_jet_sd;

    //Create branch for soft drop secondary lund plane coordinates
    vector< vector< double > > lund_coords_events_x_sd_secondary;
    vector< vector< double > > lund_coords_events_y_sd_secondary;
    vector< vector< double > > lund_delta_events_sd_secondary;
    vector< vector< double > > lund_kt_events_sd_secondary;
    vector< vector< double > > lund_z_events_sd_secondary;
    vector< vector< double > > lund_psi_events_sd_secondary;
    vector< vector< double > > lund_kappa_events_sd_secondary;
    vector< vector< double > > lund_mass_events_sd_secondary;

    vector< double > lund_coords_jet_x_sd_secondary;
    vector< double > lund_coords_jet_y_sd_secondary;
    vector< double > lund_delta_jet_sd_secondary;
    vector< double > lund_kt_jet_sd_secondary;
    vector< double > lund_z_jet_sd_secondary;
    vector< double > lund_psi_jet_sd_secondary;
    vector< double > lund_kappa_jet_sd_secondary;
    vector< double > lund_mass_jet_sd_secondary;

    //Setup branches primary plane
    auto lund_branch_x = tree->Branch("lund_coords_x", &lund_coords_events_x);
    auto lund_branch_y = tree->Branch("lund_coords_y", &lund_coords_events_y);
    auto lund_branch_delta = tree->Branch("lund_delta", &lund_delta_events);
    auto lund_branch_kt = tree->Branch("lund_kt", &lund_kt_events);
    auto lund_branch_z = tree->Branch("lund_z", &lund_z_events);
    auto lund_branch_psi = tree->Branch("lund_psi", &lund_psi_events);
    auto lund_branch_kappa = tree->Branch("lund_kappa", &lund_kappa_events);
    auto lund_branch_mass = tree->Branch("lund_mass", &lund_mass_events);
    //auto lund_branch_phi = tree->Branch("lund_phi", &lund_phi_events);

    //Setup branches secondary plane
    auto lund_branch_secondary_x = tree->Branch("lund_coords_secondary_x", &lund_coords_events_secondary_x);
    auto lund_branch_secondary_y = tree->Branch("lund_coords_secondary_y", &lund_coords_events_secondary_y);
    auto lund_branch_secondary_delta = tree->Branch("lund_delta_secondary", &lund_delta_events_secondary);
    auto lund_branch_secondary_kt = tree->Branch("lund_kt_secondary", &lund_kt_events_secondary);
    auto lund_branch_secondary_z = tree->Branch("lund_z_secondary", &lund_z_events_secondary);
    auto lund_branch_secondary_psi = tree->Branch("lund_psi_secondary", &lund_psi_events_secondary);
    auto lund_branch_secondary_kappa = tree->Branch("lund_kappa_secondary", &lund_kappa_events_secondary);
    auto lund_branch_secondary_mass = tree->Branch("lund_mass_secondary", &lund_mass_events_secondary);
    //auto lund_branch_secondary_phi = tree->Branch("lund_phi_secondary", &lund_phi_events_secondary);

    //Setup branches soft drop primary plane
    auto lund_branch_x_sd = tree->Branch("lund_coords_x_sd", &lund_coords_events_x_sd);
    auto lund_branch_y_sd = tree->Branch("lund_coords_y_sd", &lund_coords_events_y_sd);
    auto lund_branch_delta_sd = tree->Branch("lund_delta_sd", &lund_delta_events_sd);
    auto lund_branch_kt_sd = tree->Branch("lund_kt_sd", &lund_kt_events_sd);
    auto lund_branch_z_sd = tree->Branch("lund_z_sd", &lund_z_events_sd);
    auto lund_branch_psi_sd = tree->Branch("lund_psi_sd", &lund_psi_events_sd);
    auto lund_branch_kappa_sd = tree->Branch("lund_kappa_sd", &lund_kappa_events_sd);
    auto lund_branch_mass_sd = tree->Branch("lund_mass_sd", &lund_mass_events_sd);

    //Setup branches soft drop secondary plane
    auto lund_branch_x_sd_secondary = tree->Branch("lund_coords_x_sd_secondary", &lund_coords_events_x_sd_secondary);
    auto lund_branch_y_sd_secondary = tree->Branch("lund_coords_y_sd_secondary", &lund_coords_events_y_sd_secondary);
    auto lund_branch_delta_sd_secondary = tree->Branch("lund_delta_sd_secondary", &lund_delta_events_sd_secondary);
    auto lund_branch_kt_sd_secondary = tree->Branch("lund_kt_sd_secondary", &lund_kt_events_sd_secondary);
    auto lund_branch_z_sd_secondary = tree->Branch("lund_z_sd_secondary", &lund_z_events_sd_secondary);
    auto lund_branch_psi_sd_secondary = tree->Branch("lund_psi_sd_secondary", &lund_psi_events_sd_secondary);
    auto lund_branch_kappa_sd_secondary = tree->Branch("lund_kappa_sd_secondary", &lund_kappa_events_sd_secondary);
    auto lund_branch_mass_sd_secondary = tree->Branch("lund_mass_sd_secondary", &lund_mass_events_sd_secondary);

    Long64_t nevents = tree->GetEntries();
    std::cout << "Number of events: " << nevents << std::endl;

    const double sd_beta = 0;     // 0 => mMDT // -1 => "traditional" soft drop (agressive)
    const double sd_zcut = 0.1;     // typical 0.05–0.2
    const double R0   = 1.0;     // usually = jet R

    fastjet::contrib::SecondaryLund_mMDT secondary;
    fastjet::JetDefinition jet_def(fastjet::cambridge_aachen_algorithm, R0);
    fastjet::contrib::LundWithSecondary lund(jet_def, &secondary);
    fastjet::contrib::SoftDrop softdrop(sd_beta, sd_zcut, R0);
    fastjet::contrib::LundGenerator plain_lund(jet_def);

    int event_count = 0;
    while (reader.Next()) {
        lund_coords_events_x.clear();
        lund_coords_events_y.clear();
        lund_delta_events.clear();
        lund_kt_events.clear();
        lund_z_events.clear();
        lund_psi_events.clear();
        lund_kappa_events.clear();
        lund_mass_events.clear();
        //lund_phi_events.clear();

        lund_coords_events_secondary_x.clear();
        lund_coords_events_secondary_y.clear();
        lund_delta_events_secondary.clear();
        lund_kt_events_secondary.clear();
        lund_z_events_secondary.clear();
        lund_psi_events_secondary.clear();
        lund_kappa_events_secondary.clear();
        lund_mass_events_secondary.clear();
        //lund_phi_events_secondary.clear();
        //std::cout << "Processing new event" << std::endl;

        lund_coords_events_x_sd.clear();
        lund_coords_events_y_sd.clear();
        lund_delta_events_sd.clear();
        lund_kt_events_sd.clear();
        lund_z_events_sd.clear();
        lund_psi_events_sd.clear();
        lund_kappa_events_sd.clear();
        lund_mass_events_sd.clear();

        lund_coords_events_x_sd_secondary.clear();
        lund_coords_events_y_sd_secondary.clear();
        lund_delta_events_sd_secondary.clear();
        lund_kt_events_sd_secondary.clear();
        lund_z_events_sd_secondary.clear();
        lund_psi_events_sd_secondary.clear();
        lund_kappa_events_sd_secondary.clear();
        lund_mass_events_sd_secondary.clear();

        for (std::size_t ijet = 0; ijet < jet_pt->size(); ++ijet) {

            //const int& ak = (*(jetAK))[ijet];
            const auto& pts  = (*(const_pt))[ijet];
            const auto& etas = (*(const_eta))[ijet];
            const auto& phis = (*(const_phi))[ijet];
            const auto& ms   = (*(const_mass))[ijet];
            //std::cout << "Number of constituents: " << pts.size() << std::endl;
            
            std::vector<fastjet::PseudoJet> particles;
            particles.reserve(pts.size());

            for (std::size_t k = 0; k < pts.size(); ++k) {
                const double pt  = static_cast<double>(pts[k]);
                const double eta = static_cast<double>(etas[k]);
                const double phi = static_cast<double>(phis[k]);
                const double m   = std::max(0.0, static_cast<double>(ms[k]));

                const double px = pt * std::cos(phi);
                const double py = pt * std::sin(phi);
                const double pz = pt * std::sinh(eta);
                const double E  = std::sqrt(m*m + px*px + py*py + pz*pz);

                particles.emplace_back(px, py, pz, E);
            }
            
            fastjet::ClusterSequence cs(particles, jet_def);
            vector<fastjet::PseudoJet> jets = fastjet::sorted_by_pt(cs.inclusive_jets());

            lund_coords_jet_x.clear();
            lund_coords_jet_y.clear();
            lund_delta_jet.clear();
            lund_kt_jet.clear();
            lund_z_jet.clear();
            lund_psi_jet.clear();
            lund_kappa_jet.clear();
            lund_mass_jet.clear();
            //lund_phi_jet.clear();

            lund_coords_secondary_x.clear();
            lund_coords_secondary_y.clear();
            lund_delta_jet_secondary.clear();
            lund_kt_jet_secondary.clear();
            lund_z_jet_secondary.clear();
            lund_psi_jet_secondary.clear();
            lund_kappa_jet_secondary.clear();
            lund_mass_jet_secondary.clear();
            //lund_phi_jet_secondary.clear();

            lund_coords_jet_x_sd.clear();
            lund_coords_jet_y_sd.clear();
            lund_delta_jet_sd.clear();
            lund_kt_jet_sd.clear();
            lund_z_jet_sd.clear();
            lund_psi_jet_sd.clear();
            lund_kappa_jet_sd.clear();
            lund_mass_jet_sd.clear();

            lund_coords_jet_x_sd_secondary.clear();
            lund_coords_jet_y_sd_secondary.clear();
            lund_delta_jet_sd_secondary.clear();
            lund_kt_jet_sd_secondary.clear();
            lund_z_jet_sd_secondary.clear();
            lund_psi_jet_sd_secondary.clear();
            lund_kappa_jet_sd_secondary.clear();
            lund_mass_jet_sd_secondary.clear();

            /*
            if (jets.size() != 1) {
                //std::cerr << "Error: expected exactly one jet, found " << jets.size() << std::endl;
                lund_coords_events_x.push_back(lund_coords_jet_x);
                lund_coords_events_y.push_back(lund_coords_jet_y);
                continue;
            }
            */

            vector<fastjet::contrib::LundDeclustering> declusts = lund.primary(jets[0]);
            double kt_max = -1.0;
            int kt_max_index = -1;

            vector<double> psi_cum_primary;
            psi_cum_primary.reserve(declusts.size());
            vector<V3> n_primary;
            n_primary.reserve(declusts.size());

            double psi_prev = 0.0;
            bool havre_prev = false;
            V3 n_prev{0.0, 0.0, 0.0};

            for (unsigned int idecl = 0; idecl < declusts.size(); ++idecl) {

                const auto pA = declusts[idecl].harder();
                const auto pB = declusts[idecl].softer();

                V3 n_cur = unit(cross(v3(pA), v3(pB)));
                double psi_here = 0.0;
                if (!havre_prev){
                    psi_here = 0.0;
                    havre_prev = (norm(n_cur) > 0.0);
                    n_prev = n_cur;
                } else {
                    if (norm(n_cur) > 0.0){
                        psi_here = psi_prev + signed_dpsi(n_prev, n_cur, v3(pA));
                        n_prev = n_cur;
                    } else {
                        psi_here = psi_prev;
                    }
                }
                psi_prev = psi_here;
                psi_cum_primary.push_back(psi_here);
                n_primary.push_back(n_cur);

                pair<double,double> coords = declusts[idecl].lund_coordinates();
                double delta = declusts[idecl].Delta();
                double kt = declusts[idecl].kt();
                double z = declusts[idecl].z();
                double psi = declusts[idecl].psi();
                double kappa = z*declusts[idecl].Delta();
                double mass = declusts[idecl].m();
                //double phi = declusts[idecl].phi();
                //std::cout << "(" << coords.first << ", " << coords.second << ")" << std::endl;
                lund_coords_jet_x.push_back(coords.first);
                lund_coords_jet_y.push_back(coords.second);
                lund_delta_jet.push_back(delta);
                lund_kt_jet.push_back(kt);
                lund_z_jet.push_back(z);
                lund_psi_jet.push_back(psi);
                lund_kappa_jet.push_back(kappa);
                lund_mass_jet.push_back(mass);
                //lund_phi_jet.push_back(phi);
                if (kt > kt_max && z > 0.1){
                    kt_max = kt;
                    kt_max_index = idecl;
                }
            }

            double psi1 = 0.0;
            V3 n1{0.0,0.0,0.0};
            if (kt_max_index >=0){
                psi1 = psi_cum_primary[kt_max_index];
                n1 = n_primary[kt_max_index];
            }

            vector<fastjet::contrib::LundDeclustering> sec_declusts;
            if (kt_max_index >= 0) {
                const auto soft_branch = declusts[kt_max_index].softer();
                sec_declusts = plain_lund(soft_branch);
            } else {
                sec_declusts.clear();
            }

            double kt2_max = -1.0;
            int kt2_max_index = -1;

            double psi2 = 0.0;
            std::vector<double> psi_cum_secondary;
            psi_cum_secondary.reserve(sec_declusts.size());

            double psi_prev_s = psi1;
            V3 n_prev_s = n1;
            
            for (unsigned int idecl = 0; idecl < sec_declusts.size(); ++idecl) {
                double kt = sec_declusts[idecl].kt();
                double z = sec_declusts[idecl].z();
                const auto pA = sec_declusts[idecl].harder();
                const auto pB = sec_declusts[idecl].softer();

                V3 n_cur = unit(cross(v3(pA), v3(pB)));

                double psi_here = psi_prev_s;
                if (norm(n_cur) > 0.0){
                    psi_here += signed_dpsi(n_prev_s, n_cur, v3(pA));
                    n_prev_s = n_cur;
                }

                psi_prev_s = psi_here;
                psi_cum_secondary.push_back(psi_here);

                //lund_phi_jet_secondary.push_back(phi);
                if (kt > kt2_max && z > 0.1){
                    kt2_max = kt;
                    kt2_max_index = idecl;
                }
            }

            if (kt2_max_index >=0){
                psi2 = psi_cum_secondary[kt2_max_index];
            }

            double dpsi12 = wrap_angle(psi2 - psi1);
            if (kt2_max_index >=0 && kt_max_index >=0 && (*jet_pt)[ijet] > 200.0){
                delta12_hist.Fill(dpsi12);
            }

            vector<fastjet::contrib::LundDeclustering> sec_declusts_full = lund.secondary(declusts);
            for (unsigned int idecl = 0; idecl < sec_declusts_full.size(); ++idecl) {
                pair<double,double> coords = sec_declusts_full[idecl].lund_coordinates();
                double delta = sec_declusts_full[idecl].Delta();
                double kt = sec_declusts_full[idecl].kt();
                double z = sec_declusts_full[idecl].z();
                double psi = sec_declusts_full[idecl].psi();
                double kappa = z*sec_declusts_full[idecl].Delta();
                double mass = sec_declusts_full[idecl].m();
                //double phi = sec_declusts[idecl].phi();

                lund_coords_secondary_x.push_back(coords.first);
                lund_coords_secondary_y.push_back(coords.second);
                lund_delta_jet_secondary.push_back(delta);
                lund_kt_jet_secondary.push_back(kt);
                lund_z_jet_secondary.push_back(z);
                lund_psi_jet_secondary.push_back(psi);
                lund_kappa_jet_secondary.push_back(kappa);
                lund_mass_jet_secondary.push_back(mass);
            }
            
            fastjet::PseudoJet sd_jet = softdrop(jets[0]);
            vector<fastjet::contrib::LundDeclustering> sd_declusts = lund.primary(sd_jet);
            double max_kt = -1.0;
            int max_kt_index = -1;
            for (unsigned int idecl = 0; idecl < sd_declusts.size(); ++idecl) {
                pair<double,double> coords = sd_declusts[idecl].lund_coordinates();
                double delta = sd_declusts[idecl].Delta();
                double kt = sd_declusts[idecl].kt();
                double z = sd_declusts[idecl].z();
                double psi = sd_declusts[idecl].psi();
                double kappa = z*sd_declusts[idecl].Delta();
                double mass = sd_declusts[idecl].m();

                lund_coords_jet_x_sd.push_back(coords.first);
                lund_coords_jet_y_sd.push_back(coords.second);
                lund_delta_jet_sd.push_back(delta);
                lund_kt_jet_sd.push_back(kt);
                lund_z_jet_sd.push_back(z);
                lund_psi_jet_sd.push_back(psi);
                lund_kappa_jet_sd.push_back(kappa);
                lund_mass_jet_sd.push_back(mass);
            }

            vector<fastjet::contrib::LundDeclustering> sd_sec_declusts = lund.secondary(sd_declusts);


            for (unsigned int idecl = 0; idecl < sd_sec_declusts.size(); ++idecl) {
                pair<double,double> coords = sd_sec_declusts[idecl].lund_coordinates();
                double delta = sd_sec_declusts[idecl].Delta();
                double kt = sd_sec_declusts[idecl].kt();
                double z = sd_sec_declusts[idecl].z();
                double psi = sd_sec_declusts[idecl].psi();
                double kappa = z*sd_sec_declusts[idecl].Delta();
                double mass = sd_sec_declusts[idecl].m();

                lund_coords_jet_x_sd_secondary.push_back(coords.first);
                lund_coords_jet_y_sd_secondary.push_back(coords.second);
                lund_delta_jet_sd_secondary.push_back(delta);
                lund_kt_jet_sd_secondary.push_back(kt);
                lund_z_jet_sd_secondary.push_back(z);
                lund_psi_jet_sd_secondary.push_back(psi);
                lund_kappa_jet_sd_secondary.push_back(kappa);
                lund_mass_jet_sd_secondary.push_back(mass);
            }

            /*
            fastjet::PseudoJet parent1, parent2;
            while (cs.has_parents(jets[0],parent1, parent2)) {
                //std::cout << "Daughter 1: pt=" << parent1.pt() << ", eta=" << parent1.eta() << ", phi=" << parent1.phi() << std::endl;
                //std::cout << "Daughter 2: pt=" << parent2.pt() << ", eta=" << parent2.eta() << ", phi=" << parent2.phi() << std::endl;
                if (parent1.pt() < parent2.pt()) {
                    std::swap(parent1, parent2);
                }
        
                SplitVars vars = dic_var(parent1, parent2);
                double lnInvDelta = -std::log(vars.lambda_val); // ln(1/Δ)
                double lnkt       =  std::log(vars.kt);         // ln(k_t)

                std::cout << "  lambda: " << lnInvDelta //ln(1/Delta)
                          << ", k_t: " << lnkt       //ln(kt)
                          << ", mass: " << vars.mass
                          << ", z: " << vars.z
                          << ", kappa: " << vars.kappa
                          << ", psi: " << vars.psi
                          << std::endl;
                lund_coords_jet_x.push_back(lnInvDelta);
                lund_coords_jet_y.push_back(lnkt);
                lund_delta_jet.push_back(vars.lambda_val);
                lund_kt_jet.push_back(vars.kt);
                lund_z_jet.push_back(vars.z);
                lund_psi_jet.push_back(vars.psi);
                lund_kappa_jet.push_back(vars.kappa);
                lund_mass_jet.push_back(vars.mass);

                jets[0] = parent1;
            }
            */
            lund_coords_events_x.push_back(lund_coords_jet_x);
            lund_coords_events_y.push_back(lund_coords_jet_y);
            lund_delta_events.push_back(lund_delta_jet);
            lund_kt_events.push_back(lund_kt_jet);
            lund_z_events.push_back(lund_z_jet);
            lund_psi_events.push_back(lund_psi_jet);
            lund_kappa_events.push_back(lund_kappa_jet);
            lund_mass_events.push_back(lund_mass_jet);
            //lund_phi_events.push_back(lund_phi_jet);

            lund_coords_events_secondary_x.push_back(lund_coords_secondary_x);
            lund_coords_events_secondary_y.push_back(lund_coords_secondary_y);
            lund_delta_events_secondary.push_back(lund_delta_jet_secondary);
            lund_kt_events_secondary.push_back(lund_kt_jet_secondary);
            lund_z_events_secondary.push_back(lund_z_jet_secondary);
            lund_psi_events_secondary.push_back(lund_psi_jet_secondary);
            lund_kappa_events_secondary.push_back(lund_kappa_jet_secondary);
            lund_mass_events_secondary.push_back(lund_mass_jet_secondary);
            //lund_phi_events_secondary.push_back(lund_phi_jet_secondary);

            lund_coords_events_x_sd.push_back(lund_coords_jet_x_sd);
            lund_coords_events_y_sd.push_back(lund_coords_jet_y_sd);
            lund_delta_events_sd.push_back(lund_delta_jet_sd);
            lund_kt_events_sd.push_back(lund_kt_jet_sd);
            lund_z_events_sd.push_back(lund_z_jet_sd);
            lund_psi_events_sd.push_back(lund_psi_jet_sd);
            lund_kappa_events_sd.push_back(lund_kappa_jet_sd);
            lund_mass_events_sd.push_back(lund_mass_jet_sd);

            lund_coords_events_x_sd_secondary.push_back(lund_coords_jet_x_sd_secondary);
            lund_coords_events_y_sd_secondary.push_back(lund_coords_jet_y_sd_secondary);
            lund_delta_events_sd_secondary.push_back(lund_delta_jet_sd_secondary);
            lund_kt_events_sd_secondary.push_back(lund_kt_jet_sd_secondary);
            lund_z_events_sd_secondary.push_back(lund_z_jet_sd_secondary);
            lund_psi_events_sd_secondary.push_back(lund_psi_jet_sd_secondary);
            lund_kappa_events_sd_secondary.push_back(lund_kappa_jet_sd_secondary);
            lund_mass_events_sd_secondary.push_back(lund_mass_jet_sd_secondary);
        }

        //std::cout << lund_coords_events.size() << " jets processed in this event." << std::endl;
        if (event_count%1000 == 0) {
            std::cout << "Processed " << event_count << " events. Rank " << rank << std::endl;
        }
        //Now fill the branch
        lund_branch_x ->Fill();
        lund_branch_y ->Fill();
        lund_branch_delta ->Fill();
        lund_branch_kt ->Fill();
        lund_branch_z ->Fill();
        lund_branch_psi ->Fill();
        lund_branch_kappa ->Fill();
        lund_branch_mass ->Fill();
        //lund_branch_phi ->Fill();

        lund_branch_secondary_x ->Fill();
        lund_branch_secondary_y ->Fill();
        lund_branch_secondary_delta ->Fill();
        lund_branch_secondary_kt ->Fill();
        lund_branch_secondary_z ->Fill();
        lund_branch_secondary_psi ->Fill();
        lund_branch_secondary_kappa ->Fill();
        lund_branch_secondary_mass ->Fill();
        //lund_branch_secondary_phi ->Fill();

        lund_branch_x_sd ->Fill();
        lund_branch_y_sd ->Fill();
        lund_branch_delta_sd ->Fill();
        lund_branch_kt_sd ->Fill();
        lund_branch_z_sd ->Fill();
        lund_branch_psi_sd ->Fill();
        lund_branch_kappa_sd ->Fill();
        lund_branch_mass_sd ->Fill();

        lund_branch_x_sd_secondary ->Fill();
        lund_branch_y_sd_secondary ->Fill();
        lund_branch_delta_sd_secondary ->Fill();
        lund_branch_kt_sd_secondary ->Fill();
        lund_branch_z_sd_secondary ->Fill();
        lund_branch_psi_sd_secondary ->Fill();
        lund_branch_kappa_sd_secondary ->Fill();
        lund_branch_mass_sd_secondary ->Fill();

        event_count++;
    }
    tree -> Write("",TObject::kOverwrite);
    file->Close();

    //Create function to fit histogram delta12
    //TH1F 

    //Draw histogram delta12
    TCanvas* c1 = new TCanvas("c1", "Delta Psi between first and second hardest splittings", 800, 600);
    delta12_hist.GetXaxis()->SetTitle("#Delta#psi_{12} (rad)");
    delta12_hist.GetYaxis()->SetTitle("Entries");
    delta12_hist.Draw("HIST");
    c1->SaveAs(("imgs/delta_psi12_rank"+to_string(rank)+".png").c_str());

    //Save histogram delta12 into input file
    TFile* fog= TFile::Open(path, "UPDATE");
    delta12_hist.Write("delta_psi12");
    fog->Close();

    cout << "RANK " << rank << " DONE!" << endl;
    return 0;
}