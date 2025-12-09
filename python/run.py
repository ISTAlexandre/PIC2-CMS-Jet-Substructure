#mpiexec -n 4 python3 python/run.py
import subprocess
import sys
import os
from mpi4py import MPI

def sh(cmd: str):
    # Use a login shell so $(root-config ...) expands correctly
    subprocess.run(["bash", "-lc", cmd], check=True)

def main():
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # 1) Compile the C++ program

    root_folder = "root/"

    file_list = [f for f in os.listdir(root_folder) if f.endswith(".root")]

    if rank == 0:

        os.makedirs("build", exist_ok=True)
        compile_cmd = r"""
        g++ -std=c++17 -O2 \
        main/lund_plane.cpp -o build/lund_plane \
        $(fastjet-config --cxxflags) $(root-config --cflags) \
        $(fastjet-config --libs) -lfastjettools -lfastjetcontribfragile \
        $(root-config --libs)
        """

        sh(compile_cmd)

    comm.Barrier()  # Ensure all processes wait until compilation is done

    my_files = [f for i, f in enumerate(file_list) if i % size == rank]
    for fname in my_files:
        in_path = os.path.join(root_folder, fname)
        # Run the clear root script first
        subprocess.run([sys.executable, "python/clear_root.py", in_path], check=True)

        # Run the compiled program
        subprocess.run(["./build/lund_plane", in_path, str(rank)], check=True)
    
    comm.Barrier()  # Ensure all processes finish before exiting

if __name__ == "__main__":
    main()