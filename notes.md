Create environment.
```sh
conda create -n hie-mnist python=3.6
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hie-mnist
conda install pytorch-cpu==1.0.0 torchvision-cpu==0.2.1 cpuonly -c pytorch
conda install matplotlib numpy scipy dataclasses 
python -m pip install -e .
```

Verify the install.
```sh
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
```
Should print out `1.0.0 0.2.1 False`

Run the example.
```sh
cd examples
python mnist.py --epochs 10 --dim 2 --c 1.0 --no-cuda --save-model
```

TODO
- [] Reproduce the experiment on MNIST+Omniglot with [experiment.ipynb](experiment.ipynb).