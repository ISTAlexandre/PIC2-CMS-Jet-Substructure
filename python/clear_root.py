import os
import sys
import ROOT
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", help="Input ROOT filename or path")
    args = parser.parse_args()
    # Get input path
    in_arg = args.input

    # Resolve input path (try as given, then under python/)
    in_path = in_arg
    if not os.path.exists(in_path):
        candidate = os.path.join("python", in_arg)
        candidate2 = os.path.join("root", in_arg)
        if os.path.exists(candidate):
            in_path = candidate
        elif os.path.exists(candidate2):
            in_path = candidate2
        else:
            print(f"Input file not found: {in_arg}")
            sys.exit(1)

    in_file = ROOT.TFile.Open(in_path, "READ")
    if not in_file or in_file.IsZombie():
        print(f"Failed to open: {in_path}")
        sys.exit(1)

    tree = in_file.Get("jetTree")
    if not tree:
        print('TTree "jetTree" not found in file.')
        sys.exit(1)

    # Disable lund_* branches and clone only active ones
    tree.SetBranchStatus("*", 1)
    tree.SetBranchStatus("lund_*", 0)  # matches lund_coords_*, lund_delta, lund_kt, lund_z, lund_psi, lund_kappa, lund_mass

    os.makedirs("root", exist_ok=True)
    out_path = os.path.join("root", os.path.basename(in_path))
    out_file = ROOT.TFile(out_path, "RECREATE")
    if not out_file or out_file.IsZombie():
        print(f"Failed to create: {out_path}")
        sys.exit(1)

    # Clone entries of active branches only
    out_tree = tree.CloneTree(-1, "fast")
    out_tree.Write()
    out_file.Close()
    in_file.Close()

    print(f"Wrote cleaned tree to {out_path}")

if __name__ == "__main__":
    main()