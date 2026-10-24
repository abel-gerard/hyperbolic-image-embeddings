#!/bin/bash
#SBATCH --job-name=hie-mnist
#SBATCH --output=hie-mnist-%j.out
#SBATCH --partition=gpu
#SBATCH --time=08:00:00
#SBATCH --gpus=1
#SBATCH --mem=8G

set -eu
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "hie-mnist"
cd examples

nvidia-smi

python fashion.py --epochs 30 --dim 2 --c 1.0 --lr 0.0001 --save-model --run-id "$SLURM_JOB_ID"
python fashion.py --epochs 30 --dim 3 --c 1.0 --lr 0.0001 --save-model --run-id "$SLURM_JOB_ID"
# python mnist.py --epochs 30 --dim 2 --c 1.0 --lr 0.0001 --save-model --run-id "$SLURM_JOB_ID"
# python mnist.py --epochs 30 --dim 8 --c 1.0 --lr 0.0001 --save-model --run-id "$SLURM_JOB_ID"
# python mnist.py --epochs 30 --dim 16 --c 1.0 --lr 0.0001 --save-model --run-id "$SLURM_JOB_ID"
# python mnist.py --epochs 30 --dim 32 --c 1.0 --lr 0.0001 --save-model --run-id "$SLURM_JOB_ID"
