#!/bin/bash

#SBATCH --job-name=jam-spice2-fit1  ## Name of the job.
#SBATCH -A DMOBLEY_OMSF_GPU32       ## account to charge 
#SBATCH -p gpu                      ## partition name
#SBATCH -t 3-00:00:00               ## Time limit: 3 days
#SBATCH --nodes=1                   ## (-N) number of nodes to use
#SBATCH --ntasks=1                  ## (-n) number of tasks to launch
#SBATCH --cpus-per-task=1           ## number of cores the job needs
#SBATCH --gres=gpu:1                ## number of GPUs to use
#SBATCH --mem=16gb                  ## amount of memory needed. 1 cpu is billed per 9GB memory on gpu partition
#SBATCH --error=slurm-%J.err        ## error log file
#SBATCH --output=slurm-%J.out       ## output log file

set -euf -o pipefail

# dir="$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# cd $dir
cd /dfs9/dmobley-lab/joshm3/back-to-school-josh/workflow/05-fit/

mkdir spice2-fit1
pixi run --env cuda mamba env export > spice2-fit1/solved-env.yml

pixi run --env cuda python fit.py                                              \
    --tensor-ff-path ../04-parametrize/outputs/tensor_ff.pt                    \
    --tensor-tops-path ../04-parametrize/outputs/smiles_to_topologies.pt       \
    --train-dataset-paths ../02-select-data/datasets/spice2/train              \
    --test-dataset-paths ../02-select-data/datasets/spice2/test                \
    --training-config-json-path fit.jsonc                                      \
    --n-epochs 1000                                                            \
    --batch-size 500                                                           \
    --learning-rate 1e-3                                                       \
    --device gpu                                                               \
    --fitting-dir-path spice2-fit1                                             \
    --smirnoff-template ../03-generate-initial-ff/aam-ff.offxml
