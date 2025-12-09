#mpiexec -n 4 python jets_cont.py
from __future__ import print_function
import ROOT
import os
import json
try:
    import urllib2  # Py2
except ImportError:
    urllib2 = None
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

record_id = 30508
use_local = False  # set True to fall back to in/ directory
root_folder = "in/"
out_folder = "out"
if not os.path.isdir(out_folder):
    os.makedirs(out_folder)

if not os.path.isdir(out_folder):
    os.makedirs(out_folder)

def fetch_record_files(rec_id):
    api_url = "https://opendata.cern.ch/api/records/{0}".format(rec_id)
    if urllib2 is None:
        raise RuntimeError("urllib2 unavailable in this Python")
    resp = urllib2.urlopen(api_url)
    data = json.loads(resp.read())
    files = data.get("files", [])
    out = []
    for f in files:
        name = f.get("name")
        if not name.endswith(".root"):
            continue
        # Prefer EOS/XRootD path if present, else use HTTP URL
        eos_path = f.get("fullpath") or f.get("path")  # field names may vary
        if eos_path:
            # Ensure double slash after host for EOS
            url = "root://eospublic.cern.ch/{0}".format(eos_path if eos_path.startswith("/") else "/" + eos_path)
        else:
            url = f.get("url")  # falls back to HTTP
        out.append(url)
    return out

if use_local:
    file_list = [os.path.join(root_folder, f) for f in os.listdir(root_folder) if f.endswith(".root")]
else:
    file_list = fetch_record_files(record_id)

file_list = file_list[:10]  # limit for testing
# MPI partition
my_files = [f for i, f in enumerate(file_list) if i % size == rank]