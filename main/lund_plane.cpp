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

    vector< vector<double> > lund_hard_x;
    vector< vector<double> > lund_hard_y;
    vector< vector<double> > lund_hard_z;
    vector< vector<double> > lund_soft_x;
    vector< vector<double> > lund_soft_y;
    vector< vector<double> > lund_soft_z;
    vector< vector<double>> lund_pref_x;
    vector< vector<double>> lund_pref_y;
    vector< vector<double>> lund_pref_z;

    vector<double> lund_hard_x_jet;
    vector<double> lund_hard_y_jet;
    vector<double> lund_hard_z_jet;
    vector<double> lund_soft_x_jet;
    vector<double> lund_soft_y_jet;
    vector<double> lund_soft_z_jet;
    vector<double> lund_pref_x_jet;
    vector<double> lund_pref_y_jet;
    vector<double> lund_pref_z_jet;

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

    //Create branch for first primary splitting lund plane coordinates
    /*
    vector< vector< double > > lund_coords_events_sd_first_x;
    vector< vector< double > > lund_coords_events_sd_first_y;
    vector< vector< double > > lund_delta_events_sd_first;
    vector< vector< double > > lund_kt_events_sd_first;
    vector< vector< double > > lund_z_events_sd_first;
    vector< vector< double > > lund_psi_events_sd_first;
    vector< vector< double > > lund_kappa_events_sd_first;
    vector< vector< double > > lund_mass_events_sd_first;

    vector< double > lund_coords_jet_sd_first_x;
    vector< double > lund_coords_jet_sd_first_y;
    vector< double > lund_delta_jet_sd_first;
    vector< double > lund_kt_jet_sd_first;
    vector< double > lund_z_jet_sd_first;
    vector< double > lund_psi_jet_sd_first;
    vector< double > lund_kappa_jet_sd_first;
    vector< double > lund_mass_jet_sd_first;
    */

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

    auto lund_hard_x_branch = tree->Branch("lund_hard_x", &lund_hard_x);
    auto lund_hard_y_branch = tree->Branch("lund_hard_y", &lund_hard_y);
    auto lund_hard_z_branch = tree->Branch("lund_hard_z", &lund_hard_z);
    auto lund_soft_x_branch = tree->Branch("lund_soft_x", &lund_soft_x);
    auto lund_soft_y_branch = tree->Branch("lund_soft_y", &lund_soft_y);
    auto lund_soft_z_branch = tree->Branch("lund_soft_z", &lund_soft_z);
    auto lund_pref_x_branch = tree->Branch("lund_pref_x", &lund_pref_x);
    auto lund_pref_y_branch = tree->Branch("lund_pref_y", &lund_pref_y);
    auto lund_pref_z_branch = tree->Branch("lund_pref_z", &lund_pref_z);

    //Setup branches first soft drop primary splitting plane
    /*
    auto lund_branch_x_sd_first = tree->Branch("lund_coords_x_sd_first", &lund_coords_events_sd_first_x);
    auto lund_branch_y_sd_first = tree->Branch("lund_coords_y_sd_first", &lund_coords_events_sd_first_y);
    auto lund_branch_delta_sd_first = tree->Branch("lund_delta_sd_first", &lund_delta_events_sd_first);
    auto lund_branch_kt_sd_first = tree->Branch("lund_kt_sd_first", &lund_kt_events_sd_first);
    auto lund_branch_z_sd_first = tree->Branch("lund_z_sd_first", &lund_z_events_sd_first);
    auto lund_branch_psi_sd_first = tree->Branch("lund_psi_sd_first", &lund_psi_events_sd_first);
    auto lund_branch_kappa_sd_first = tree->Branch("lund_kappa_sd_first", &lund_kappa_events_sd_first);
    auto lund_branch_mass_sd_first = tree->Branch("lund_mass_sd_first", &lund_mass_events_sd_first);
    */
    Long64_t nevents = tree->GetEntries();
    std::cout << "Number of events: " << nevents << std::endl;

    const double sd_beta = 0;     // 0 => mMDT // -1 => "traditional" soft drop (agressive)
    const double sd_zcut = 0.1;     // typical 0.05–0.2
    const double R0   = 1.0;     // usually = jet R

    fastjet::contrib::SecondaryLund_mMDT secondary;
    fastjet::JetDefinition jet_def(fastjet::cambridge_aachen_algorithm, R0);
    fastjet::contrib::LundWithSecondary lund(jet_def, &secondary);
    fastjet::contrib::SoftDrop softdrop(sd_beta, sd_zcut, R0);
    //fastjet::contrib::LundGenerator plain_lund(jet_def);

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

        /*
        lund_coords_events_sd_first_x.clear();
        lund_coords_events_sd_first_y.clear();
        lund_delta_events_sd_first.clear();
        lund_kt_events_sd_first.clear();
        lund_z_events_sd_first.clear();
        lund_psi_events_sd_first.clear();
        lund_kappa_events_sd_first.clear();
        lund_mass_events_sd_first.clear();
        */

        lund_hard_x.clear();
        lund_hard_y.clear();
        lund_hard_z.clear();
        lund_soft_x.clear();
        lund_soft_y.clear();
        lund_soft_z.clear();
        lund_pref_x.clear();
        lund_pref_y.clear();
        lund_pref_z.clear();

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

            lund_hard_x_jet.clear();
            lund_hard_y_jet.clear();
            lund_hard_z_jet.clear();
            lund_soft_x_jet.clear();
            lund_soft_y_jet.clear();
            lund_soft_z_jet.clear();
            lund_pref_x_jet.clear();
            lund_pref_y_jet.clear();
            lund_pref_z_jet.clear();

            /*
            lund_coords_jet_sd_first_x.clear();
            lund_coords_jet_sd_first_y.clear();
            lund_delta_jet_sd_first.clear();
            lund_kt_jet_sd_first.clear();
            lund_z_jet_sd_first.clear();
            lund_psi_jet_sd_first.clear();
            lund_kappa_jet_sd_first.clear();
            lund_mass_jet_sd_first.clear();
            */

            /*
            if (jets.size() != 1) {
                //std::cerr << "Error: expected exactly one jet, found " << jets.size() << std::endl;
                lund_coords_events_x.push_back(lund_coords_jet_x);
                lund_coords_events_y.push_back(lund_coords_jet_y);
                continue;
            }
            */

            vector<fastjet::contrib::LundDeclustering> declusts = lund.primary(jets[0]);
            int first_primary = declusts.empty() ? -1 : 0;

            for (unsigned int idecl = 0; idecl < declusts.size(); ++idecl) {
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
            }

            vector<fastjet::contrib::LundDeclustering> sec_declusts = lund.secondary(declusts);

            for (unsigned int idecl = 0; idecl < sec_declusts.size(); ++idecl) {
                pair<double,double> coords = sec_declusts[idecl].lund_coordinates();
                double delta = sec_declusts[idecl].Delta();
                double kt = sec_declusts[idecl].kt();
                double z = sec_declusts[idecl].z();
                double psi = sec_declusts[idecl].psi();
                double kappa = z*sec_declusts[idecl].Delta();
                double mass = sec_declusts[idecl].m();
                //double phi = sec_declusts[idecl].phi();

                lund_coords_secondary_x.push_back(coords.first);
                lund_coords_secondary_y.push_back(coords.second);
                lund_delta_jet_secondary.push_back(delta);
                lund_kt_jet_secondary.push_back(kt);
                lund_z_jet_secondary.push_back(z);
                lund_psi_jet_secondary.push_back(psi);
                lund_kappa_jet_secondary.push_back(kappa);
                lund_mass_jet_secondary.push_back(mass);
                //lund_phi_jet_secondary.push_back(phi);
            }
            
            fastjet::PseudoJet sd_jet = softdrop(jets[0]);
            vector<fastjet::contrib::LundDeclustering> sd_declusts = lund.primary(sd_jet);
            
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

                fastjet::PseudoJet p1, p2;
                p1 = sd_declusts[idecl].harder();
                p2 = sd_declusts[idecl].softer();
                //Get vector normal
                double cross_x = p1.py()*p2.pz() - p1.pz()*p2.py();
                double cross_y = p1.pz()*p2.px() - p1.px()*p2.pz();
                double cross_z = p1.px()*p2.py() - p1.py()*p2.px();
                lund_hard_x_jet.push_back(cross_x);
                lund_hard_y_jet.push_back(cross_y);
                lund_hard_z_jet.push_back(cross_z);
            }
            
            /*
            if (!sd_declusts.empty()){
                auto soft_branch = sd_declusts[0].softer();

                vector<fastjet::contrib::LundDeclustering> sb_declusts = plain_lund(soft_branch);

                for (unsigned int idecl = 0; idecl < sb_declusts.size(); ++idecl) {
                    pair<double,double> coords = sb_declusts[idecl].lund_coordinates();
                    double delta = sb_declusts[idecl].Delta();
                    double kt = sb_declusts[idecl].kt();
                    double z = sb_declusts[idecl].z();
                    double psi = sb_declusts[idecl].psi();
                    double kappa = z*sb_declusts[idecl].Delta();
                    double mass = sb_declusts[idecl].m();

                    lund_coords_jet_sd_first_x.push_back(coords.first);
                    lund_coords_jet_sd_first_y.push_back(coords.second);
                    lund_delta_jet_sd_first.push_back(delta);
                    lund_kt_jet_sd_first.push_back(kt);
                    lund_z_jet_sd_first.push_back(z);
                    lund_psi_jet_sd_first.push_back(psi);
                    lund_kappa_jet_sd_first.push_back(kappa);
                    lund_mass_jet_sd_first.push_back(mass);
                }
            }
            */


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

                fastjet::PseudoJet p1, p2;
                p1 = sd_sec_declusts[idecl].harder();
                p2 = sd_sec_declusts[idecl].softer();
                //Get vector normal
                double cross_x = p1.py()*p2.pz() - p1.pz()*p2.py();
                double cross_y = p1.pz()*p2.px() - p1.px()*p2.pz();
                double cross_z = p1.px()*p2.py() - p1.py()*p2.px();
                lund_soft_x_jet.push_back(cross_x);
                lund_soft_y_jet.push_back(cross_y);
                lund_soft_z_jet.push_back(cross_z);

                lund_pref_x_jet.push_back(p1.px());
                lund_pref_y_jet.push_back(p1.py());
                lund_pref_z_jet.push_back(p1.pz());
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

            lund_hard_x.push_back(lund_hard_x_jet);
            lund_hard_y.push_back(lund_hard_y_jet);
            lund_hard_z.push_back(lund_hard_z_jet);
            lund_soft_x.push_back(lund_soft_x_jet);
            lund_soft_y.push_back(lund_soft_y_jet);
            lund_soft_z.push_back(lund_soft_z_jet);
            lund_pref_x.push_back(lund_pref_x_jet);
            lund_pref_y.push_back(lund_pref_y_jet);
            lund_pref_z.push_back(lund_pref_z_jet);

            /*
            lund_coords_events_sd_first_x.push_back(lund_coords_jet_sd_first_x);
            lund_coords_events_sd_first_y.push_back(lund_coords_jet_sd_first_y);
            lund_delta_events_sd_first.push_back(lund_delta_jet_sd_first);
            lund_kt_events_sd_first.push_back(lund_kt_jet_sd_first);
            lund_z_events_sd_first.push_back(lund_z_jet_sd_first);
            lund_psi_events_sd_first.push_back(lund_psi_jet_sd_first);
            lund_kappa_events_sd_first.push_back(lund_kappa_jet_sd_first);
            lund_mass_events_sd_first.push_back(lund_mass_jet_sd_first);
            */

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
        
        /*
        lund_branch_x_sd_first ->Fill();
        lund_branch_y_sd_first ->Fill();
        lund_branch_delta_sd_first ->Fill();
        lund_branch_kt_sd_first ->Fill();
        lund_branch_z_sd_first ->Fill();
        lund_branch_psi_sd_first ->Fill();
        lund_branch_kappa_sd_first ->Fill();
        lund_branch_mass_sd_first ->Fill();
        */

        lund_hard_x_branch ->Fill();
        lund_hard_y_branch ->Fill();
        lund_hard_z_branch ->Fill();
        lund_soft_x_branch ->Fill();
        lund_soft_y_branch ->Fill();
        lund_soft_z_branch ->Fill();
        lund_pref_x_branch ->Fill();
        lund_pref_y_branch ->Fill();
        lund_pref_z_branch ->Fill();

        event_count++;
    }
    tree -> Write("",TObject::kOverwrite);
    file->Close();
    cout << "RANK " << rank << " DONE!" << endl;
    return 0;
}