# -*- coding: utf-8 -*-
from __future__ import print_function
import ROOT
import os
import json

try:
    unicode  # Python 2
except NameError:
    unicode = str

out_folder = "out"
merged_path = os.path.join(out_folder, "merged.root")

output_files = sorted(
    os.path.join(out_folder, f)
    for f in os.listdir(out_folder)
    if f.startswith("out_") and f.endswith(".root")
)

if len(output_files) == 0:
    raise RuntimeError("No output ROOT files found to merge in: {}".format(out_folder))

# Prefer calling hadd with a list to avoid shell quoting issues
import subprocess

cmd = ["hadd", "-f", merged_path] + output_files
print("Merging {} files -> {}".format(len(output_files), merged_path))
subprocess.check_call(cmd)