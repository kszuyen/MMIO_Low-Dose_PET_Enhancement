import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os, argparse, sys, json
from utils import *
from unet import UNET
from dataset import NTUH_dataset
import numpy as np
import nibabel as nib
from tqdm import tqdm

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

def load_config(project_name, fold, case, batch_size):
    cfg = dotdict(dict())
    cfg.project_name = project_name
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

    root_dir = os.path.join(results_file, cfg.project_name, f"fold{fold}")
    models_dir = os.path.join(root_dir, "models_file")
    cfg.best_ckpt_dir = os.path.join(models_dir, f"{cfg.mod_name}_best.pth")

    return cfg

def output_single_nii(np_dict, output_dir):
    Z = max(np_dict)
    WH = np_dict[Z].shape
    np_arr = np.zeros(shape=(WH[0], WH[1], Z+1))
    for z in range(0, Z+1):
        np_arr[:, :, z] = np_dict[z]
    NII_IMG = nib.Nifti1Image(np_arr, affine=np.eye(4))
    nib.save(NII_IMG, filename=output_dir)

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument("--project_name", "-P",help="select [EarlyFrame] or [LowDose]", type=str)
    parser.add_argument("--json_file", type=str)
    parser.add_argument("--results_file", type=str)
    args = parser.parse_args()
    project_name = args.project_name
    json_file = args.json_file
    results_file = args.results_file

    output_dir = os.path.join(DIR_PATH, project_name+'_output_3dnifti')
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
            for patient_fullname in testing_patients:
                patient = int(patient_fullname[-3:])
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
                all_mod_3dnparr_dict[int(patient)]['ct'][int(slc)] = data[0].detach().cpu().numpy()
                all_mod_3dnparr_dict[int(patient)]['mr'][int(slc)] = data[1].detach().cpu().numpy()
                all_mod_3dnparr_dict[int(patient)]['pt'][int(slc)] = data[2].detach().cpu().numpy()
                all_mod_3dnparr_dict[int(patient)]['gt'][int(slc)] = gt[0].detach().cpu().numpy()

            for patient in all_mod_3dnparr_dict:
                output_single_nii(all_mod_3dnparr_dict[patient]['ct'], os.path.join(output_dir, f"PIB{str(patient).zfill(3)}", "CT.nii"))
                output_single_nii(all_mod_3dnparr_dict[patient]['mr'], os.path.join(output_dir, f"PIB{str(patient).zfill(3)}", "MR.nii"))
                output_single_nii(all_mod_3dnparr_dict[patient]['pt'], os.path.join(output_dir, f"PIB{str(patient).zfill(3)}", "PT.nii"))
                output_single_nii(all_mod_3dnparr_dict[patient]['gt'], os.path.join(output_dir, f"PIB{str(patient).zfill(3)}", "GT.nii"))
            ### 

            for case in range(1, 5):

                cfg = load_config(project_name, fold, int(case), batch_size=1)
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

                    for patient_fullname in testing_patients:
                        patient = int(patient_fullname[-3:])
                        pred_3dnparr_dict[patient] = dict()
                        
                    for batch_idx, (batch_data, batch_gt, batch_patient, batch_slc) in enumerate(test_loader):

                        mini_batch_size = batch_gt.shape[0]
                        batch_data, batch_gt = batch_data.to(device, dtype=torch.float), batch_gt.to(device, dtype=torch.float)
                        output = model(batch_data)

                        for i, (p, s) in enumerate(zip(list(batch_patient), list(batch_slc))):
                            pred_3dnparr_dict[int(p)][int(s)] = reverse_normalize(scaler=min_max_scaler[2])(output[i][0]).detach().cpu().numpy()

                    for patient in pred_3dnparr_dict:
                        output_single_nii(pred_3dnparr_dict[patient], os.path.join(output_dir, f"PIB{str(patient).zfill(3)}", f"PRED_Case{str(case)}.nii"))