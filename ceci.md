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