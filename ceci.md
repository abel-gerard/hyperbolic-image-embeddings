```sh
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
source $HOME/miniconda3/etc/profile.d/conda.sh
rm Miniconda3-latest-Linux-x86_64.sh

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

```sh
conda create -y --prefix $GLOBALSCRATCH/hie-mnist python=3.6
conda activate $GLOBALSCRATCH/hie-mnist
```

```sh
pip install https://download.pytorch.org/whl/cu100/torch-1.0.0-cp36-cp36m-linux_x86_64.whl
conda install -y cudatoolkit=10.0 -c pytorch
pip install torchvision==0.2.1

pip install numpy==1.16.6
pip install dataclasses==0.8
pip install matplotlib==3.3.4
pip install scipy==1.4.1
```

```sh
python -m pip install -e .
```

## PyTorch 1.0.0 on Lyra's RTX 6000 Ada

The environment above specifies the PyTorch 1.0.0 CUDA 10.0 wheel. Lyra nodes have NVIDIA RTX 6000 Ada GPUs (compute capability 8.9), much newer than this software stack, so test compatibility rather than assume it. In the first reported Slurm test, PyTorch detected the GPU and a 1024x1024 multiplication completed; this did not reproduce the original hang. Seeing the GPU name alone still does not prove every kernel or workload is correct. A container can package software, but it cannot add GPU-architecture support to a wheel or replace the host's GPU driver.

To pinpoint where execution stalls, run a small test in a Slurm GPU job, not on a login node. Save this as `torch100-smoke.sbatch`:

```sh
#!/bin/bash
#SBATCH --job-name=torch100-smoke
#SBATCH --output=torch100-smoke-%j.out
#SBATCH --time=00:05:00
#SBATCH --gpus=1

set -eu
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$GLOBALSCRATCH/hie-mnist"

nvidia-smi
python -u - <<'PY'
import torch

print("PyTorch:", torch.__version__, flush=True)
print("PyTorch CUDA build:", torch.version.cuda, flush=True)
print("CUDA available:", torch.cuda.is_available(), flush=True)
print("GPU:", torch.cuda.get_device_name(0), flush=True)
print("Compute capability:", torch.cuda.get_device_capability(0), flush=True)

x = torch.randn(1024, 1024).cuda()
print("Tensor copied to GPU", flush=True)
torch.cuda.synchronize()
print("Copy synchronized", flush=True)

y = torch.mm(x, x)
print("Matrix multiplication submitted", flush=True)
torch.cuda.synchronize()
print("Matrix multiplication completed:", y[0, 0].item(), flush=True)

y_gpu = y.cpu()
x_cpu = x.cpu()
y_cpu = torch.mm(x_cpu, x_cpu)
print("GPU output abs max:", y_gpu.abs().max().item(), flush=True)
print("Max abs difference vs CPU:", (y_gpu - y_cpu).abs().max().item(), flush=True)
PY
```

Submit with `sbatch torch100-smoke.sbatch` and inspect the output file. The last printed checkpoint identifies whether the stall is during transfer, synchronization, or multiplication. The CPU comparison checks that the GPU returned plausible numerical results; a single output element (including `0.0`) is not enough to establish correctness. Keep the short job time limit so a stalled GPU call does not occupy a resource indefinitely.

In a later run using PyTorch 1.0.0 with `torch.version.cuda` reported as `10.0.130`, the test detected the RTX 6000 Ada but stalled before printing `Tensor copied to GPU`. That places the hang at the initial `.cuda()` allocation/copy, before matrix multiplication. A separate run in the `hie-mnist-gpu` Conda environment reported CUDA `9.0.176` and completed the 1024x1024 multiplication, but a CPU comparison showed GPU output abs max `0.0` and maximum absolute difference `158.46621704101562`: that result is incorrect. This is a different environment from the `hie-mnist` CUDA 10.0 setup above, so the exact installed wheel/environment matters:

```sh
python -c 'import torch; print(torch.__file__); print(torch.__version__, torch.version.cuda)'
python -m pip show torch
conda list | grep -E '^(torch|pytorch|cudatoolkit)[[:space:]]'
```

`nvidia-smi`'s `CUDA Version: 13.0` is the driver's advertised CUDA compatibility, not the CUDA version PyTorch was built against. Neither the CUDA 9.0 nor CUDA 10.0 build tested here provides a working GPU path on Ada: one returns incorrect values, and the other stalls during the initial device copy. An Apptainer image does not by itself resolve this. NVIDIA's PyTorch 19.01 container is also PyTorch 1.0/CUDA 10.0, and its release notes list Pascal, Volta, and Turing GPUs; Ada support is not established.

A more promising CÉCI alternative is the Dragon2 GPU queue: the cluster page lists two nodes with NVIDIA Tesla V100 GPUs (Volta), and NVIDIA's PyTorch 19.01 release notes include Volta in the supported GPU families. Recreate the PyTorch 1.0.0/CUDA 10.0 environment there and run the same numerical smoke test before moving the full workload. The Dragon2 page lists the generic GPU resource as `gpu`; confirm the account and exact Slurm request with the cluster documentation. If Dragon2 access is unavailable, ask CÉCI support (`ceci-support@ulb.be`) about a validated legacy-compatible GPU setup. Include the test outputs and `nvidia-smi` details. If Lyra Ada is mandatory, a source/compatibility port or relaxing the exact PyTorch version may be necessary.

- [Lyra cluster specifications and support](https://www.ceci-hpc.be/clusters/lyra/)
- [Dragon2 cluster specifications and support](https://www.ceci-hpc.be/clusters/dragon2/)
- [NVIDIA CUDA GPU compute capabilities](https://developer.nvidia.com/cuda-gpus)
- [CÉCI Slurm tutorial](https://support.ceci-hpc.be/doc/SubmittingJobs/SlurmTutorial/)
- [CÉCI Conda guidance](https://support.ceci-hpc.be/doc/UsingSoftwareAndLibraries/InstallingSoftwareByYourself/conda/)
- [CÉCI Singularity/Apptainer guidance](https://support.ceci-hpc.be/doc/UsingSoftwareAndLibraries/Singularity/)
- [NVIDIA PyTorch 19.01 release notes](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel_19-01.html)