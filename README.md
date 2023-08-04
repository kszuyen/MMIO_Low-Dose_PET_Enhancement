# MMIO_Low-Dose_PET_Enhancement project

## CLONE THIS REPO
```
git clone https://github.com/kszuyen/MMIO_Low-Dose_PET_Enhancement.git
cd MMIO_Low-Dose_PET_Enhancement
```

## CREATE ENVIRONMENT
```
conda create -n <env_name> python=3.8
conda activate <env_name>
```
Install **PyTorch** from [PyTorch website](https://pytorch.org/get-started/locally/)    
example for our MMIO A100 server:
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
Install other required packages
```
pip install -r requirements.txt
```

## START TRAINING

### Modify parameters in `train.sh`
- DATA_DIR: the directory for the coregistered **ntuh dataset** (should include mr, ct, and pt 3d nifti images)
- P_NAME: Select "LowDose" or "EarlyFrame" or "LowDose_with_T1"
- TOTAL_FOLD: How much fold you want to split the patients
- JSON_FILE: only change this if you want a specific path for the split json file. Default: "${TOTAL_FOLD}fold.json"
- NUM_EPOCHS: max number of epochs for each case
- LEARNING_RATE: learning rate
- CUDA: you can specify the gpu you want to train with. Default: 0.
```
bash train.sh
```


