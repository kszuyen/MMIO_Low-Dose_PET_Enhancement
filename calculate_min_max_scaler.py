import torch.nn as nn
import torch
from dataset import NTUH_dataset
from tqdm import tqdm

def calculate_min_max_scaler(root_dir, fold):
    train_dataset = NTUH_dataset(
        root_dir=root_dir,
        dataset_type="train",
        min_max_scaler=None,
        DataAugmentation=False,
        fold=fold
    )

    ct_min = torch.min(train_dataset[0][0][0,:,:])
    ct_max = torch.max(train_dataset[0][0][0,:,:])
    mr_min = torch.min(train_dataset[0][0][1,:,:])
    mr_max = torch.max(train_dataset[0][0][1,:,:])
    pet_min = min(torch.min(train_dataset[0][0][2,:,:]), torch.min(train_dataset[0][1]))
    pet_max = max(torch.max(train_dataset[0][0][2,:,:]), torch.max(train_dataset[0][1]))
    # gt_min = torch.min(train_dataset[0][1])
    # gt_max = torch.max(train_dataset[0][1])
    

    print("Calculating global min/max:")
    for i in tqdm(range(1, len(train_dataset))):
        ct_min = min(ct_min, torch.min(train_dataset[i][0][0,:,:]))
        ct_max = max(ct_max, torch.max(train_dataset[i][0][0,:,:]))
        mr_min = min(mr_min, torch.min(train_dataset[i][0][1,:,:]))
        mr_max = max(mr_max, torch.max(train_dataset[i][0][1,:,:]))
        pet_min = min(pet_min, torch.min(train_dataset[i][0][2,:,:]), torch.min(train_dataset[i][1]))
        pet_max = max(pet_max, torch.max(train_dataset[i][0][2,:,:]), torch.max(train_dataset[i][1]))
        # gt_min = min(gt_min, torch.min(train_dataset[i][1]))
        # gt_max = max(gt_max, torch.max(train_dataset[i][1]))
    min_max = [
        (ct_min.item(), ct_max.item()),
        (mr_min.item(), mr_max.item()),
        (pet_min.item(), pet_max.item()),
        # (gt_min.item(), gt_max.item())
    ]
    return min_max


if __name__ == "__main__":
    for fold in tqdm(range(1, 11)):

        root_dir = f"/home/kszuyen/MMIO_Low-Dose_PET_Enhancement/2d_data_LowDose_with_T1_fold{fold}"

        min_max = calculate_min_max_scaler(root_dir)
        print("[(ct_min, ct_max), (mr_min, mr_max), (pt_min, pt_max)]: ")
        print(min_max)
