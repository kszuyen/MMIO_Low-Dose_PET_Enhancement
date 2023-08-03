# MMIO_Low-Dose_PET_Enhancement project

## CREATE ENVIRONMENT
```
conda create -n <env_name> python=3.8
conda activate <env_name>
```
Install PyTorch from [PyTorch website](https://pytorch.org/get-started/locally/)    
example for our MMIO A100 server:
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
Install other required packages
```
pip install -r requirements.txt
```

