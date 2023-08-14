import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os, argparse, sys, json
from utils import *
from unet import UNET
from dataset import NTUH_dataset
import numpy as np
from tqdm import tqdm
import SimpleITK as sitk

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

def load_config(project_name, project_results_dir, fold, case, batch_size):
    cfg = dotdict(dict())
    cfg.project_name = project_name
    cfg.project_results_dir = project_results_dir

    cfg.data_dir = os.path.join(DIR_PATH, f"2d_data_{project_name}_fold{fold}")
    cfg.case = case
    cfg.out_channels = 1
    cfg.batch_size = batch_size

    if case == 1:
        cfg.mod_name = "pt_only"
        cfg.in_channels = 1
    elif case == 2:
        cfg.mod_name = "ct_pt"
        cfg.in_channels = 2
    elif case == 3:
        cfg.mod_name = "mr_pt"
        cfg.in_channels = 2
    elif case == 4:
        cfg.mod_name = "ct_mr_pt"
        cfg.in_channels = 3
    else:
        print("""Case error, please select case 1, 2, 3 or 4:
        1: PT only
        2: CT & PT
        3: MR & PT
        4: CT, MR & PT
        """)
        sys.exit()

    # root_dir = os.path.join(DIR_PATH, "results", cfg.project_name, f"fold{fold}")
    models_dir = os.path.join(project_results_dir, f"fold{fold}", "models_file")
    cfg.best_ckpt_dir = os.path.join(models_dir, f"{cfg.mod_name}_best.pth")

    return cfg

def output_single_nii(np_dict, output_dir, image_info, Z_slice=91, pad_value=0):
    Z = max(np_dict)
    WH = np_dict[Z].shape
    np_arr = np.zeros(shape=(WH[0], WH[1], Z+1))
    np_arr = np.full((WH[0], WH[1], Z_slice),fill_value=pad_value, dtype=float)
    for z in np_dict:
        np_arr[:, :, z] = np_dict[z]
    # NII_IMG = nib.Nifti1Image(np_arr, affine=np.eye(4))
    NII_IMG = sitk.GetImageFromArray(np_arr)
    NII_IMG.SetOrigin(image_info[0])
    NII_IMG.SetDirection(image_info[1])
    NII_IMG.SetSpacing(image_info[2])
    sitk.WriteImage(NII_IMG, fileName=output_dir)

def calculate_min_value(root_dir):
    train_dataset = NTUH_dataset(
        root_dir=root_dir,
        dataset_type="train",
        min_max_scaler=None,
        DataAugmentation=False,
    )
    valid_dataset = NTUH_dataset(
        root_dir=root_dir,
        dataset_type="val",
        min_max_scaler=None,
        DataAugmentation=False,
    )
    test_dataset = NTUH_dataset(
        root_dir=root_dir,
        dataset_type="test",
        min_max_scaler=None,
        DataAugmentation=False
    )
    ct_min = torch.min(train_dataset[0][0][0,:,:])
    mr_min = torch.min(train_dataset[0][0][1,:,:])
    pet_min = min(torch.min(train_dataset[0][0][2,:,:]), torch.min(train_dataset[0][1]))

    for dataset in [train_dataset, valid_dataset, test_dataset]:
        for i in range(0, len(dataset)):
            ct_min = min(ct_min, torch.min(dataset[i][0][0,:,:]))
            mr_min = min(mr_min, torch.min(dataset[i][0][1,:,:]))
            pet_min = min(pet_min, torch.min(dataset[i][0][2,:,:]), torch.min(dataset[i][1]))

    return ct_min.item(), mr_min.item(), pet_min.item()

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument("--project_name", "-P",help="select [EarlyFrame] or [LowDose]", type=str)
    parser.add_argument("--json_file", type=str)
    parser.add_argument("--project_results_dir", type=str)
    parser.add_argument("--max_z_slice", type=int, default=91)

    args = parser.parse_args()
    project_name = args.project_name
    json_file = args.json_file
    project_results_dir = args.project_results_dir
    max_z_slice = args.max_z_slice

    print("Preparing....")
    ct_pad_value, mr_pad_value, pt_pad_value = calculate_min_value(os.path.join(DIR_PATH, f"2d_data_{project_name}_fold1"))
    
    output_dir = os.path.join(DIR_PATH, +f'output_3dnifti_{project_name}')
    make_dir(output_dir)
    with open(os.path.join(DIR_PATH, json_file) , 'r') as openfile:
        # Reading from json file
        json_object = json.load(openfile)
        TOTAL_FOLD = len(json_object)
        
        for fold in tqdm(range(1, 11)):
            testing_index = (fold + TOTAL_FOLD - 2) % TOTAL_FOLD
            testing_patients = json_object[testing_index]
            for patient in testing_patients:
                make_dir(os.path.join(output_dir, patient))

            ###
            ### output mr, ct, pt, gt
            all_mod_3dnparr_dict = dict()
            for patient in testing_patients:
                # patient = int(patient_fullname[-3:])
                all_mod_3dnparr_dict[patient] = dict()
                for mod in ['ct', 'mr', 'pt', 'gt']:
                    all_mod_3dnparr_dict[patient][mod] = dict()

            test_dataset = NTUH_dataset(
                root_dir=os.path.join(DIR_PATH, f"2d_data_{project_name}_fold{fold}"),
                dataset_type="test",
                case=4,
                min_max_scaler=None,
            )

            for idx, (data, gt, patient, slc) in enumerate(test_dataset):
                all_mod_3dnparr_dict[patient]['ct'][int(slc)] = data[0].detach().cpu().numpy()
                all_mod_3dnparr_dict[patient]['mr'][int(slc)] = data[1].detach().cpu().numpy()
                all_mod_3dnparr_dict[patient]['pt'][int(slc)] = data[2].detach().cpu().numpy()
                all_mod_3dnparr_dict[patient]['gt'][int(slc)] = gt[0].detach().cpu().numpy()

            for patient in all_mod_3dnparr_dict:
                patient_image_info = np.load(os.path.join(DIR_PATH, f"2d_data_{project_name}_fold{fold}", "image_info", f"{patient}.npy"), allow_pickle=True)
                for mod, output_name, pad_value in zip(['ct', 'mr', 'pt', 'gt'], ["CT.nii", "MR.nii", "PT.nii", "GT.nii"], [ct_pad_value, mr_pad_value, pt_pad_value, pt_pad_value]):
                    output_single_nii(
                        np_dict=all_mod_3dnparr_dict[patient][mod], 
                        output_dir=os.path.join(output_dir, patient, output_name),
                        image_info=patient_image_info,
                        Z_slice=max_z_slice,
                        pad_value=pad_value
                    )
            ### 

            for case in range(1, 5):

                cfg = load_config(project_name, project_results_dir, fold, int(case), batch_size=1)
                if os.path.exists(cfg.best_ckpt_dir): # if chechpoint exists
                    # load model and checkpoint
                    model = UNET(in_ch=cfg.in_channels, out_ch=cfg.out_channels).to(device)
                    ckpt = torch.load(cfg.best_ckpt_dir, map_location=device)
                    model.load_state_dict(ckpt['model'])
                    min_max_scaler = ckpt['min_max_scaler']

                    test_dataset = NTUH_dataset(
                        root_dir=cfg.data_dir,
                        dataset_type="test",
                        case=cfg.case,
                        min_max_scaler=min_max_scaler,
                    )
                    test_loader = DataLoader(
                        dataset=test_dataset,
                        batch_size=cfg.batch_size,
                        shuffle=False
                    )
                    model.eval()

                    pred_3dnparr_dict = dict()

                    for patient in testing_patients:
                        pred_3dnparr_dict[patient] = dict()
                        
                    for batch_idx, (batch_data, batch_gt, batch_patient, batch_slc) in enumerate(test_loader):

                        mini_batch_size = batch_gt.shape[0]
                        batch_data, batch_gt = batch_data.to(device, dtype=torch.float), batch_gt.to(device, dtype=torch.float)
                        output = model(batch_data)

                        for i, (p, s) in enumerate(zip(list(batch_patient), list(batch_slc))):
                            pred_3dnparr_dict[p][int(s)] = reverse_normalize(scaler=min_max_scaler[2])(output[i][0]).detach().cpu().numpy()

                    for patient in pred_3dnparr_dict:
                        patient_image_info = np.load(os.path.join(DIR_PATH, f"2d_data_{project_name}_fold{fold}", "image_info", f"{patient}.npy"), allow_pickle=True)
                        output_single_nii(
                            np_dict=pred_3dnparr_dict[patient], 
                            output_dir=os.path.join(output_dir, patient, f"PRED_Case{str(case)}.nii"),
                            image_info=patient_image_info,
                            Z_slice=max_z_slice,
                            pad_value=pt_pad_value
                        )