#!/bin/bash
#SBATCH --job-name=mnist-hie
#SBATCH --output=%j_mnist.out
#SBATCH --error=%j_mnist.err
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

#SBATCH --partition=batch
#SBATCH --gpus=1

module purge

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate $GLOBALSCRATCH/hie-mnist
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

cd $HOME/hyperbolic-image-embeddings/examples
python -u mnist.py --epochs 10 --dim 2 --c 1.0 --save-model
