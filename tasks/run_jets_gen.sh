#!/bin/bash
#SBATCH -p lipq
#SBATCH --mem=8G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --array=0-15          # 16 tasks = 16 parallel workers
                               # adjust upper bound to number of files - 1
#SBATCH --job-name=jets_gen
#SBATCH --output=logs/slurm-%A_%a.out   # %A = job id, %a = task id

# INPUT = jets_gen_slurm.py
# INPUT = txt_ML/
# OUTPUT = out_ML/

set -euo pipefail

JOBDIR="$PWD"
echo "Task $SLURM_ARRAY_TASK_ID / $SLURM_ARRAY_TASK_COUNT on $(hostname)"

source /cvmfs/cms.cern.ch/cmsset_default.sh
export SCRAM_ARCH=slc7_amd64_gcc700

cd "$JOBDIR"
scram project CMSSW CMSSW_10_6_30
cd CMSSW_10_6_30/src
cmsenv

cd "$JOBDIR"
mkdir -p out_ML logs

# SLURM_ARRAY_TASK_ID and SLURM_ARRAY_TASK_COUNT are automatically
# set by SLURM — jets_gen.py reads them instead of MPI rank/size
python -u jets_gen_slurm.py