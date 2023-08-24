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

## SPLIT PATIENTS

Select a project_name: **LowDose** or **EarlyFrame**, and run
```
python split.py -P <project_name>
```
This will split the original 3d nifti data into 2d data for training.
### You can also add arguments:
- total_fold: How many folds you want for the training, and it will split the patients into N groups. Default: 10.
- split_json_fild: Specify the path to an already split json file if you want to split the patients into the same groups as the json file.

## START TRAINING

### Modify parameters in `train.sh`
- P_NAME: Select the project_name
- NUM_EPOCHS: max number of epochs for each case
- LEARNING_RATE: learning rate
- CUDA: you can specify the gpu you want to train with. Default: 0.
```
bash train.sh
```

## INFERENCE
Select the project_name in **inference_all.sh** and run
```
bash inference_all.sh
```
This will output the 2d png results, 3d nifti results, and the ssim, psnr scores for your training results.

