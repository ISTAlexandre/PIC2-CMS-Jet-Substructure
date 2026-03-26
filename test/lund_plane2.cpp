/*
g++ -std=c++17 -O2 \
  main/lund_plane2.cpp -o build/lund_plane2 \
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
#include <memory>
#include <string>

#include <algorithm>
#include <limits>

using namespace std;

struct Vec3 {
    double x, y, z;
};

static inline Vec3 make_vec3(const fastjet::PseudoJet& p) {
    return {p.px(), p.py(), p.pz()};
}

static inline Vec3 cross3(const Vec3& a, const Vec3& b) {
    return {
        a.y*b.z - a.z*b.y,
        a.z*b.x - a.x*b.z,
        a.x*b.y - a.y*b.x
    };
}

static inline double dot3(const Vec3& a, const Vec3& b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

static inline double norm3(const Vec3& a) {
    return std::sqrt(dot3(a,a));
}

static inline Vec3 unit3(const Vec3& a) {
    double n = norm3(a);
    if (n <= 0.0) return {0.0, 0.0, 0.0};
    return {a.x/n, a.y/n, a.z/n};
}

static inline double wrap_to_pi(double x) {
    while (x <= -M_PI) x += 2.0* M_PI;
    while (x >   M_PI) x -= 2.0* M_PI;
    return x;
}


// Signed angle between two plane normals n_prev -> n_curr,
// sign fixed by (n_prev x n_curr) . pA_curr (harder prong of current split)
static double signed_plane_angle(const Vec3& n_prev,
    const Vec3& n_curr,
    const fastjet::PseudoJet& pA_curr) {
Vec3 pA = make_vec3(pA_curr);

double c = std::clamp(dot3(n_prev, n_curr), -1.0, 1.0);
double ang = std::acos(c);

Vec3 cr = cross3(n_prev, n_curr);
double s = dot3(cr, pA);

// If nearly collinear normals, sign is numerically unstable; angle ~ 0 (or pi)
if (norm3(cr) < 1e-14) return ang;

return (s >= 0.0) ? ang : -ang;
}

struct PsiChain {
    std::vector<Vec3> normals;      // splitting-plane normals
    std::vector<char> valid_normal; // 1 if normal is valid
    std::vector<double> psi;        // cumulative article-style psi
};

// Build normals for a declustering list
static PsiChain build_normals_only(const std::vector<fastjet::contrib::LundDeclustering>& decls) {
    PsiChain out;
    size_t n = decls.size();
    out.normals.resize(n);
    out.valid_normal.assign(n, 0);
    out.psi.assign(n, 0.0);

    for (size_t i = 0; i < n; ++i) {
        Vec3 a = make_vec3(decls[i].harder());
        Vec3 b = make_vec3(decls[i].softer());
        Vec3 nvec = cross3(a, b);
        double nn = norm3(nvec);
        if (nn > 1e-14) {
            out.normals[i] = unit3(nvec);
            out.valid_normal[i] = 1;
        } else {
            out.normals[i] = {0.0, 0.0, 0.0};
            out.valid_normal[i] = 0;
        }
    }
    return out;
}

// Article-style cumulative psi on a PRIMARY chain.
// Convention: psi[0] = 0 (arbitrary offset; cancels in differences)
static void fill_primary_psi(const std::vector<fastjet::contrib::LundDeclustering>& decls, PsiChain& chain) {
if (decls.empty()) return;
chain.psi[0] = 0.0;

for (size_t i = 1; i < decls.size(); ++i) {
    chain.psi[i] = chain.psi[i-1];
    if (!chain.valid_normal[i-1] || !chain.valid_normal[i]) continue;

    double dpsi = signed_plane_angle(chain.normals[i-1], chain.normals[i], decls[i].harder());
    chain.psi[i] += dpsi;
}
}

// Article-style cumulative psi on the SECONDARY chain, anchored to the selected primary split.
// parent_normal = normal of selected primary split
// parent_psi    = psi of selected primary split
static void fill_secondary_psi_from_parent(const std::vector<fastjet::contrib::LundDeclustering>& decls,
                  PsiChain& chain,
                  const Vec3& parent_normal,
                  double parent_psi,
                  bool parent_normal_valid) {
if (decls.empty()) return;

// First secondary split is referenced to the selected primary split plane
chain.psi[0] = parent_psi;
if (parent_normal_valid && chain.valid_normal[0]) {
chain.psi[0] += signed_plane_angle(parent_normal, chain.normals[0], decls[0].harder());
}

// Then recurse within the secondary chain
for (size_t i = 1; i < decls.size(); ++i) {
chain.psi[i] = chain.psi[i-1];
if (!chain.valid_normal[i-1] || !chain.valid_normal[i]) continue;

double dpsi = signed_plane_angle(chain.normals[i-1], chain.normals[i], decls[i].harder());
chain.psi[i] += dpsi;
}
}

static int select_maxkt_with_zcut(const std::vector<fastjet::contrib::LundDeclustering>& decls, double zcut) {
int idx = -1;
double best_kt = -1.0;
for (int i = 0; i < (int)decls.size(); ++i) {
if (decls[i].z() > zcut && decls[i].kt() > best_kt) {
best_kt = decls[i].kt();
idx = i;
}
}
return idx;
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

    fastjet::contrib::SecondaryLund_mMDT secondary(sd_zcut);
    fastjet::JetDefinition jet_def(fastjet::cambridge_aachen_algorithm, R0);
    fastjet::contrib::LundWithSecondary lund(jet_def, &secondary);
    fastjet::contrib::SoftDrop softdrop(sd_beta, sd_zcut, R0);
    fastjet::contrib::LundGenerator plain_lund(jet_def);

    int event_count = 0;

    // Create ROOT branches for output
    vector< vector<double> > lund_coords_events_x; //Primary plane coordinates
    vector< vector<double> > lund_coords_events_y;
    vector< vector<double> > lund_delta_events;
    vector< vector<double> > lund_kt_events;
    vector< vector<double> > lund_z_events;
    vector< vector<double> > lund_psi_events;
    vector< vector<double> > lund_kappa_events;
    vector< vector<double> > lund_mass_events;

    vector< vector<double> > lund_coords_events_secondary_x; //Secondary plane coordinates
    vector< vector<double> > lund_coords_events_secondary_y;
    vector< vector<double> > lund_delta_events_secondary;
    vector< vector<double> > lund_kt_events_secondary;
    vector< vector<double> > lund_z_events_secondary;
    vector< vector<double> > lund_psi_events_secondary;
    vector< vector<double> > lund_kappa_events_secondary;
    vector< vector<double> > lund_mass_events_secondary;

    vector< double > lund_psi12_events; //Delta psi between first and second hardest splits
    vector< double > lund_delta_psi12_events_sd; //Delta psi between first and second hardest splits on soft drop plane

    vector< vector<double> > lund_coords_events_x_sd; //Soft drop primary plane coordinates
    vector< vector<double> > lund_coords_events_y_sd;
    vector< vector<double> > lund_delta_events_sd;
    vector< vector<double> > lund_kt_events_sd;
    vector< vector<double> > lund_z_events_sd;
    vector< vector<double> > lund_psi_events_sd;
    vector< vector<double> > lund_kappa_events_sd;
    vector< vector<double> > lund_mass_events_sd;

    vector< vector<double> > lund_coords_events_x_sd_secondary; //Soft drop secondary plane coordinates
    vector< vector<double> > lund_coords_events_y_sd_secondary;
    vector< vector<double> > lund_delta_events_sd_secondary;
    vector< vector<double> > lund_kt_events_sd_secondary;
    vector< vector<double> > lund_z_events_sd_secondary;
    vector< vector<double> > lund_psi_events_sd_secondary;
    vector< vector<double> > lund_kappa_events_sd_secondary;
    vector< vector<double> > lund_mass_events_sd_secondary;

    // Setup ROOT branches
    auto lund_branch_x = tree->Branch("lund_coords_x", &lund_coords_events_x);
    auto lund_branch_y = tree->Branch("lund_coords_y", &lund_coords_events_y);
    auto lund_branch_delta = tree->Branch("lund_delta", &lund_delta_events);
    auto lund_branch_kt = tree->Branch("lund_kt", &lund_kt_events);
    auto lund_branch_z = tree->Branch("lund_z", &lund_z_events);
    auto lund_branch_psi = tree->Branch("lund_psi", &lund_psi_events);
    auto lund_branch_kappa = tree->Branch("lund_kappa", &lund_kappa_events);
    auto lund_branch_mass = tree->Branch("lund_mass", &lund_mass_events);

    auto lund_branch_secondary_x = tree->Branch("lund_coords_secondary_x", &lund_coords_events_secondary_x);
    auto lund_branch_secondary_y = tree->Branch("lund_coords_secondary_y", &lund_coords_events_secondary_y);
    auto lund_branch_secondary_delta = tree->Branch("lund_delta_secondary", &lund_delta_events_secondary);
    auto lund_branch_secondary_kt = tree->Branch("lund_kt_secondary", &lund_kt_events_secondary);
    auto lund_branch_secondary_z = tree->Branch("lund_z_secondary", &lund_z_events_secondary);
    auto lund_branch_secondary_psi = tree->Branch("lund_psi_secondary", &lund_psi_events_secondary);
    auto lund_branch_secondary_kappa = tree->Branch("lund_kappa_secondary", &lund_kappa_events_secondary);
    auto lund_branch_secondary_mass = tree->Branch("lund_mass_secondary", &lund_mass_events_secondary);

    auto lund_branch_x_sd = tree->Branch("lund_coords_x_sd", &lund_coords_events_x_sd);
    auto lund_branch_y_sd = tree->Branch("lund_coords_y_sd", &lund_coords_events_y_sd);
    auto lund_branch_delta_sd = tree->Branch("lund_delta_sd", &lund_delta_events_sd);
    auto lund_branch_kt_sd = tree->Branch("lund_kt_sd", &lund_kt_events_sd);
    auto lund_branch_z_sd = tree->Branch("lund_z_sd", &lund_z_events_sd);
    auto lund_branch_psi_sd = tree->Branch("lund_psi_sd", &lund_psi_events_sd);
    auto lund_branch_kappa_sd = tree->Branch("lund_kappa_sd", &lund_kappa_events_sd);
    auto lund_branch_mass_sd = tree->Branch("lund_mass_sd", &lund_mass_events_sd);

    auto lund_branch_x_sd_secondary = tree->Branch("lund_coords_x_sd_secondary", &lund_coords_events_x_sd_secondary);
    auto lund_branch_y_sd_secondary = tree->Branch("lund_coords_y_sd_secondary", &lund_coords_events_y_sd_secondary);
    auto lund_branch_delta_sd_secondary = tree->Branch("lund_delta_sd_secondary", &lund_delta_events_sd_secondary);
    auto lund_branch_kt_sd_secondary = tree->Branch("lund_kt_sd_secondary", &lund_kt_events_sd_secondary);
    auto lund_branch_z_sd_secondary = tree->Branch("lund_z_sd_secondary", &lund_z_events_sd_secondary);
    auto lund_branch_psi_sd_secondary = tree->Branch("lund_psi_sd_secondary", &lund_psi_events_sd_secondary);
    auto lund_branch_kappa_sd_secondary = tree->Branch("lund_kappa_sd_secondary", &lund_kappa_events_sd_secondary);
    auto lund_branch_mass_sd_secondary = tree->Branch("lund_mass_sd_secondary", &lund_mass_events_sd_secondary);

    auto lund_branch_psi12 = tree->Branch("lund_psi12", &lund_psi12_events);
    auto lund_branch_psi12_sd = tree->Branch("lund_psi12_sd", &lund_delta_psi12_events_sd);


     // Working vectors for jet-level declustering (reset for each jet)
     vector<double> lund_coords_jet_x;
     vector<double> lund_coords_jet_y;
     vector<double> lund_delta_jet;
     vector<double> lund_kt_jet;
     vector<double> lund_z_jet;
     vector<double> lund_psi_jet;
     vector<double> lund_kappa_jet;
     vector<double> lund_mass_jet;
    
    while (reader.Next()) {

        // Clear branches
        lund_coords_events_x.clear();
        lund_coords_events_y.clear();
        lund_delta_events.clear();
        lund_kt_events.clear();
        lund_z_events.clear();
        lund_psi_events.clear();
        lund_kappa_events.clear();
        lund_mass_events.clear();

        lund_coords_events_secondary_x.clear();
        lund_coords_events_secondary_y.clear();
        lund_delta_events_secondary.clear();
        lund_kt_events_secondary.clear();
        lund_z_events_secondary.clear();
        lund_psi_events_secondary.clear();
        lund_kappa_events_secondary.clear();
        lund_mass_events_secondary.clear();

        lund_psi12_events.clear();
        lund_delta_psi12_events_sd.clear();

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

        //For jet clustering (more than 1 jet per event)
        for (size_t ijet=0; ijet < jet_pt->size(); ++ijet) {
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

            // Cluster the constituents into jets using the specified jet definition
            fastjet::ClusterSequence cs(constituents, jet_def);
            vector<fastjet::PseudoJet> jets = fastjet::sorted_by_pt(cs.inclusive_jets());

            if (jets.size() == 0 || jets.size() > 1) {
                lund_coords_events_x.push_back({});
                lund_coords_events_y.push_back({});
                lund_delta_events.push_back({});
                lund_kt_events.push_back({});
                lund_z_events.push_back({});
                lund_psi_events.push_back({});
                lund_kappa_events.push_back({});
                lund_mass_events.push_back({});

                lund_coords_events_secondary_x.push_back({});
                lund_coords_events_secondary_y.push_back({});
                lund_delta_events_secondary.push_back({});
                lund_kt_events_secondary.push_back({});
                lund_z_events_secondary.push_back({});
                lund_psi_events_secondary.push_back({});
                lund_kappa_events_secondary.push_back({});
                lund_mass_events_secondary.push_back({});

                lund_coords_events_x_sd.push_back({});
                lund_coords_events_y_sd.push_back({});
                lund_delta_events_sd.push_back({});
                lund_kt_events_sd.push_back({});
                lund_z_events_sd.push_back({});
                lund_psi_events_sd.push_back({});
                lund_kappa_events_sd.push_back({});
                lund_mass_events_sd.push_back({});

                lund_coords_events_x_sd_secondary.push_back({});
                lund_coords_events_y_sd_secondary.push_back({});
                lund_delta_events_sd_secondary.push_back({});
                lund_kt_events_sd_secondary.push_back({});
                lund_z_events_sd_secondary.push_back({});
                lund_psi_events_sd_secondary.push_back({});
                lund_kappa_events_sd_secondary.push_back({});
                lund_mass_events_sd_secondary.push_back({});

                lund_psi12_events.push_back(std::numeric_limits<double>::quiet_NaN());
                lund_delta_psi12_events_sd.push_back(std::numeric_limits<double>::quiet_NaN());
                cout << "Warning: expected exactly one jet, found " << jets.size() << " in event " << event_count << ", jet index " << ijet << endl;
                continue; // Sanity check
            }

            // Reset vectors for primnary declustering
            lund_coords_jet_x.clear();
            lund_coords_jet_y.clear();
            lund_delta_jet.clear();
            lund_kt_jet.clear();
            lund_z_jet.clear();
            lund_psi_jet.clear();
            lund_kappa_jet.clear();
            lund_mass_jet.clear();

            //Decluster the leading jet with the Lund generator
            vector<fastjet::contrib::LundDeclustering> declusts = lund.primary(jets[0]);

            for (unsigned int idecl = 0; idecl < declusts.size(); ++idecl) {
                pair<double,double> coords = declusts[idecl].lund_coordinates();
                double delta = declusts[idecl].Delta();
                double kt = declusts[idecl].kt();
                double z = declusts[idecl].z();
                double psi = declusts[idecl].psi();
                double kappa = z*declusts[idecl].Delta();
                double mass = declusts[idecl].m();

                lund_coords_jet_x.push_back(coords.first);
                lund_coords_jet_y.push_back(coords.second);
                lund_delta_jet.push_back(delta);
                lund_kt_jet.push_back(kt);
                lund_z_jet.push_back(z);
                lund_psi_jet.push_back(psi);
                lund_kappa_jet.push_back(kappa);
                lund_mass_jet.push_back(mass);
            }

            //Add to event-level vectors
            lund_coords_events_x.push_back(lund_coords_jet_x);
            lund_coords_events_y.push_back(lund_coords_jet_y);
            lund_delta_events.push_back(lund_delta_jet);
            lund_kt_events.push_back(lund_kt_jet);
            lund_z_events.push_back(lund_z_jet);
            lund_psi_events.push_back(lund_psi_jet);
            lund_kappa_events.push_back(lund_kappa_jet);
            lund_mass_events.push_back(lund_mass_jet);

            // Reset vectors for secondary declustering
            lund_coords_jet_x.clear();
            lund_coords_jet_y.clear();
            lund_delta_jet.clear();
            lund_kt_jet.clear();
            lund_z_jet.clear();
            lund_psi_jet.clear();
            lund_kappa_jet.clear();
            lund_mass_jet.clear();

            double delta_psi12 = numeric_limits<double>::quiet_NaN();
            bool has_delta_psi12 = false;

            int i1 = select_maxkt_with_zcut(declusts, sd_zcut);

            if (i1 >= 0){
                PsiChain prim_chain = build_normals_only(declusts);
                fill_primary_psi(declusts, prim_chain);

                // Build the ARTICLE secondary plane: decluster the softer branch of selected primary split
                vector<fastjet::contrib::LundDeclustering> sec_declusts = plain_lund(declusts[i1].softer());

                for (unsigned int idecl = 0; idecl < sec_declusts.size(); ++idecl) {
                    pair<double,double> coords = sec_declusts[idecl].lund_coordinates();
                    double delta = sec_declusts[idecl].Delta();
                    double kt = sec_declusts[idecl].kt();
                    double z = sec_declusts[idecl].z();
                    double psi = sec_declusts[idecl].psi();
                    double kappa = z*sec_declusts[idecl].Delta();
                    double mass = sec_declusts[idecl].m();

                    lund_coords_jet_x.push_back(coords.first);
                    lund_coords_jet_y.push_back(coords.second);
                    lund_delta_jet.push_back(delta);
                    lund_kt_jet.push_back(kt);
                    lund_z_jet.push_back(z);
                    lund_psi_jet.push_back(psi);
                    lund_kappa_jet.push_back(kappa);
                    lund_mass_jet.push_back(mass);
                }

                int i2 = select_maxkt_with_zcut(sec_declusts, sd_zcut);
                if (i2 >= 0) {
                    // Build article-style psi on the secondary chain, anchored to selected primary split
                    PsiChain sec_chain = build_normals_only(sec_declusts);
                    fill_secondary_psi_from_parent(sec_declusts,
                                                   sec_chain,
                                                   prim_chain.normals[i1],
                                                   prim_chain.psi[i1],
                                                   prim_chain.valid_normal[i1]);
            
                    delta_psi12 = wrap_to_pi(sec_chain.psi[i2] - prim_chain.psi[i1]);
                    has_delta_psi12 = true;
                }
            }

            //Add to event-level vectors
            lund_coords_events_secondary_x.push_back(lund_coords_jet_x);
            lund_coords_events_secondary_y.push_back(lund_coords_jet_y);
            lund_delta_events_secondary.push_back(lund_delta_jet);
            lund_kt_events_secondary.push_back(lund_kt_jet);
            lund_z_events_secondary.push_back(lund_z_jet);
            lund_psi_events_secondary.push_back(lund_psi_jet);
            lund_kappa_events_secondary.push_back(lund_kappa_jet);
            lund_mass_events_secondary.push_back(lund_mass_jet);

            if (has_delta_psi12) {
                lund_psi12_events.push_back(delta_psi12);
            } else {
                lund_psi12_events.push_back(numeric_limits<double>::quiet_NaN());
            }

            /*
            if (has_delta_psi12) {
                cout << "delta_psi12 = " << delta_psi12 << "\n";
            }
            */

            // Reset vectors for soft drop primary declustering
            lund_coords_jet_x.clear();
            lund_coords_jet_y.clear();
            lund_delta_jet.clear();
            lund_kt_jet.clear();
            lund_z_jet.clear();
            lund_psi_jet.clear();
            lund_kappa_jet.clear();
            lund_mass_jet.clear();

            vector<fastjet::contrib::LundDeclustering> sd_declusts = lund.primary(softdrop(jets[0]));
            for (unsigned int idecl = 0; idecl < sd_declusts.size(); ++idecl) {
                pair<double,double> coords = sd_declusts[idecl].lund_coordinates();
                double delta = sd_declusts[idecl].Delta();
                double kt = sd_declusts[idecl].kt();
                double z = sd_declusts[idecl].z();
                double psi = sd_declusts[idecl].psi();
                double kappa = z*sd_declusts[idecl].Delta();
                double mass = sd_declusts[idecl].m();

                lund_coords_jet_x.push_back(coords.first);
                lund_coords_jet_y.push_back(coords.second);
                lund_delta_jet.push_back(delta);
                lund_kt_jet.push_back(kt);
                lund_z_jet.push_back(z);
                lund_psi_jet.push_back(psi);
                lund_kappa_jet.push_back(kappa);
                lund_mass_jet.push_back(mass);
            }

            //Add to event-level vectors
            lund_coords_events_x_sd.push_back(lund_coords_jet_x);
            lund_coords_events_y_sd.push_back(lund_coords_jet_y);
            lund_delta_events_sd.push_back(lund_delta_jet);
            lund_kt_events_sd.push_back(lund_kt_jet);
            lund_z_events_sd.push_back(lund_z_jet);
            lund_psi_events_sd.push_back(lund_psi_jet);
            lund_kappa_events_sd.push_back(lund_kappa_jet);
            lund_mass_events_sd.push_back(lund_mass_jet);

            // Reset vectors for soft drop secondary declustering
            lund_coords_jet_x.clear();
            lund_coords_jet_y.clear();
            lund_delta_jet.clear();
            lund_kt_jet.clear();
            lund_z_jet.clear();
            lund_psi_jet.clear();
            lund_kappa_jet.clear();
            lund_mass_jet.clear();

            double delta_psi12_sd = numeric_limits<double>::quiet_NaN();
            bool has_delta_psi12_sd = false;

            int i1_sd = select_maxkt_with_zcut(sd_declusts, sd_zcut);

            if (i1_sd >= 0){
                PsiChain prim_chain_sd = build_normals_only(sd_declusts);
                fill_primary_psi(sd_declusts, prim_chain_sd);

                // Build the ARTICLE secondary plane: decluster the softer branch of selected primary split
                vector<fastjet::contrib::LundDeclustering> sec_declusts_sd = plain_lund(sd_declusts[i1_sd].softer());

                for (unsigned int idecl = 0; idecl < sec_declusts_sd.size(); ++idecl) {
                    pair<double,double> coords = sec_declusts_sd[idecl].lund_coordinates();
                    double delta = sec_declusts_sd[idecl].Delta();
                    double kt = sec_declusts_sd[idecl].kt();
                    double z = sec_declusts_sd[idecl].z();
                    double psi = sec_declusts_sd[idecl].psi();
                    double kappa = z*sec_declusts_sd[idecl].Delta();
                    double mass = sec_declusts_sd[idecl].m();

                    lund_coords_jet_x.push_back(coords.first);
                    lund_coords_jet_y.push_back(coords.second);
                    lund_delta_jet.push_back(delta);
                    lund_kt_jet.push_back(kt);
                    lund_z_jet.push_back(z);
                    lund_psi_jet.push_back(psi);
                    lund_kappa_jet.push_back(kappa);
                    lund_mass_jet.push_back(mass);
                }

                int i2_sd = select_maxkt_with_zcut(sec_declusts_sd, sd_zcut);
                if (i2_sd >= 0) {
                    // Build article-style psi on the secondary chain, anchored to selected primary split
                    PsiChain sec_chain_sd = build_normals_only(sec_declusts_sd);
                    fill_secondary_psi_from_parent(sec_declusts_sd,
                                                   sec_chain_sd,
                                                   prim_chain_sd.normals[i1_sd],
                                                   prim_chain_sd.psi[i1_sd],
                                                   prim_chain_sd.valid_normal[i1_sd]);
                    delta_psi12_sd = wrap_to_pi(sec_chain_sd.psi[i2_sd] - prim_chain_sd.psi[i1_sd]);
                    has_delta_psi12_sd = true;
                }
            }

            //Add to event-level vectors
            lund_coords_events_x_sd_secondary.push_back(lund_coords_jet_x);
            lund_coords_events_y_sd_secondary.push_back(lund_coords_jet_y);
            lund_delta_events_sd_secondary.push_back(lund_delta_jet);
            lund_kt_events_sd_secondary.push_back(lund_kt_jet);
            lund_z_events_sd_secondary.push_back(lund_z_jet);
            lund_psi_events_sd_secondary.push_back(lund_psi_jet);
            lund_kappa_events_sd_secondary.push_back(lund_kappa_jet);
            lund_mass_events_sd_secondary.push_back(lund_mass_jet);

            if (has_delta_psi12_sd) {
                lund_delta_psi12_events_sd.push_back(delta_psi12_sd);
            } else {
                lund_delta_psi12_events_sd.push_back(numeric_limits<double>::quiet_NaN());
            }

        }
    if (event_count%1000 == 0) {
        cout << "Rank " << rank << ": processed " << event_count << " events of " << nevents << endl;
    }
    event_count++;
    // Fill the branches for this event
    lund_branch_x ->Fill();
    lund_branch_y ->Fill();
    lund_branch_delta ->Fill();
    lund_branch_kt ->Fill();
    lund_branch_z ->Fill();
    lund_branch_psi ->Fill();
    lund_branch_kappa ->Fill();
    lund_branch_mass ->Fill();
    lund_branch_secondary_x ->Fill();
    lund_branch_secondary_y ->Fill();
    lund_branch_secondary_delta ->Fill();
    lund_branch_secondary_kt ->Fill();
    lund_branch_secondary_z ->Fill();
    lund_branch_secondary_psi ->Fill();
    lund_branch_secondary_kappa ->Fill();
    lund_branch_secondary_mass ->Fill();

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

    lund_branch_psi12 ->Fill();
    lund_branch_psi12_sd ->Fill();
    }
    tree->Write("", TObject::kOverwrite);
    file->Write("", TObject::kOverwrite);
    file->Close();

    cout << "Rank " << rank << ": finished processing " << event_count << " events.\n";
}