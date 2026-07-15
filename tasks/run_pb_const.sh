#!/bin/bash
#SBATCH -p lipq
#SBATCH --mem=8G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --array=0-7          # 16 tasks = 16 parallel workers
                               # adjust upper bound to number of files - 1
#SBATCH --job-name=reco_const
#SBATCH --output=logs/slurm-%A_%a.out   # %A = job id, %a = task id

# INPUT = reco_const_slurm.py
# INPUT = Cert_181530-183126_HI7TeV_PromptReco_Collisions11_JSON.txt
# INPUT = txt/
# OUTPUT = out/

set -euo pipefail

JOBDIR="$PWD"
echo "Task $SLURM_ARRAY_TASK_ID / $SLURM_ARRAY_TASK_COUNT on $(hostname)"

source /cvmfs/cms.cern.ch/cmsset_default.sh
export SCRAM_ARCH=slc5_amd64_gcc434

cd "$JOBDIR"
scram project CMSSW CMSSW_4_4_7
cd CMSSW_4_4_7/src
cmsenv

cd "$JOBDIR"
mkdir -p out logs

# SLURM_ARRAY_TASK_ID and SLURM_ARRAY_TASK_COUNT are automatically
# set by SLURM — jets_gen.py reads them instead of MPI rank/size
python -u reco_const_slurm.py