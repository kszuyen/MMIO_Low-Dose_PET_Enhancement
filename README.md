# MMIO_Low-Dose_PET_Enhancement

## CREATE ENVIRONMENT
```
conda create -n <env_name> python=3.8
conda activate <env_name>
```
install PyTorch from PyTorch website (https://pytorch.org/get-started/locally/)
ps: for mmio a100 server, must use "pip install" instead of "conda install"
ex:
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
install other required packages
```
pip install -r requirements
```

## SPLIT PATIENTS
`python split.py`
