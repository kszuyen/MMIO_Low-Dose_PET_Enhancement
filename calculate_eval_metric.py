import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os, argparse, sys, csv, json
from utils import *
from unet import UNET
from dataset import NTUH_dataset
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import numpy as np
from tqdm import tqdm

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

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument("--project_name", "-P",help="select [EarlyFrame] or [LowDose]", type=str)
    parser.add_argument("--json_file", type=str)
    parser.add_argument("--project_results_dir", type=str)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()
    project_name = args.project_name
    json_file = args.json_file
    project_results_dir = args.project_results_dir
    batch_size = args.batch_size

    patient_list = []
    orig_scores = [dict() for _ in range(2)]
    new_psnr = [dict() for _ in range(4)]
    new_ssim = [dict() for _ in range(4)]

    with open(os.path.join(DIR_PATH, json_file) , 'r') as openfile:
        # Reading from json file
        json_object = json.load(openfile)
        TOTAL_FOLD = len(json_object)

        for fold in tqdm(range(1, TOTAL_FOLD + 1)):
            testing_index = (fold + TOTAL_FOLD - 2) % TOTAL_FOLD
            testing_patients = json_object[testing_index]
            CALCULATED = False

            for case in range(1, 5):
                cfg = load_config(project_name, project_results_dir, fold, int(case), batch_size=batch_size)
                if os.path.exists(cfg.best_ckpt_dir): # if chechpoint exists
                    # load model and checkpoint
                    model = UNET(in_ch=cfg.in_channels, out_ch=cfg.out_channels).to(device)
                    ckpt = torch.load(cfg.best_ckpt_dir, map_location=device)
                    model.load_state_dict(ckpt['model'])
                    min_max_scaler = ckpt['min_max_scaler']
                    # print("Best epoch:", ckpt['epoch'])

                    test_dataset = NTUH_dataset(
                        root_dir=cfg.data_dir,
                        dataset_type="test",
                        case=cfg.case,
                        min_max_scaler=min_max_scaler
                    )
                    test_loader = DataLoader(
                        dataset=test_dataset,
                        batch_size=cfg.batch_size,
                        shuffle=False
                    )
                    model.eval()
                    for batch_idx, (batch_data, batch_gt, batch_patient, batch_slc) in enumerate(test_loader):
                        mini_batch_size = batch_gt.shape[0]
                        batch_data, batch_gt = batch_data.to(device, dtype=torch.float), batch_gt.to(device, dtype=torch.float)
                        output = model(batch_data)

                        for i in range(mini_batch_size):
                            patient = int(batch_patient[i])

                            """ original score """
                            if not CALCULATED: # only need to calculate once for each patient

                                pet = reverse_normalize(scaler=min_max_scaler[-1])(batch_data[i][-1]).detach().cpu().numpy()
                                pet_cor = reverse_normalize(scaler=min_max_scaler[-1])(batch_gt[i][0]).detach().cpu().numpy()
                                orig_psnr = peak_signal_noise_ratio(pet, pet_cor, data_range=min_max_scaler[-1][1]-min_max_scaler[-1][0])
                                orig_ssim = structural_similarity(pet, pet_cor, data_range=min_max_scaler[-1][1]-min_max_scaler[-1][0])
                                
                                if patient in orig_scores[0]:
                                    orig_scores[0][patient].append(orig_psnr)
                                    orig_scores[1][patient].append(orig_ssim)
                                else:
                                    orig_scores[0][patient] = [orig_psnr]
                                    orig_scores[1][patient] = [orig_ssim]  

                            """  new score  """
                            pred = reverse_normalize(scaler=min_max_scaler[-1])(output[i][0]).detach().cpu().numpy()
                            gt = reverse_normalize(scaler=min_max_scaler[-1])(batch_gt[i][0]).detach().cpu().numpy()

                            cur_psnr = peak_signal_noise_ratio(pred, gt, data_range=min_max_scaler[-1][1]-min_max_scaler[-1][0])
                            cur_ssim = structural_similarity(pred, gt, data_range=min_max_scaler[-1][1]-min_max_scaler[-1][0])
                            
                            if patient in new_psnr[case-1]:
                                new_psnr[case-1][patient].append(cur_psnr)
                                new_ssim[case-1][patient].append(cur_ssim)
                            else:
                                new_psnr[case-1][patient] = [cur_psnr]
                                new_ssim[case-1][patient] = [cur_ssim]

                    CALCULATED = True
                            
        orig_list = [[] for _ in range(2)]
        psnr_list = [[] for _ in range(4)]
        ssim_list = [[] for _ in range(4)]
        for c in range(4):
            for patient in {key:new_psnr[c][key] for key in sorted(new_psnr[c].keys())}:
                if c == 0:
                    patient_list.append("PIB_"+str(patient))
                    orig_list[0].append(np.mean(orig_scores[0][patient]))
                    orig_list[1].append(np.mean(orig_scores[1][patient]))
                psnr_list[c].append(np.mean(new_psnr[c][patient]))
                ssim_list[c].append(np.mean(new_ssim[c][patient]))
                    
        all_score = []
        all_score.append(patient_list)
        for scores in [orig_list, psnr_list, ssim_list]:
            for i in range(len(scores)):
                all_score.append(scores[i])

        csv_file = os.path.join(DIR_PATH, f"{project_name}_scores.csv")
        with open(csv_file, "w") as f:
            writer = csv.writer(f)
            row_name = ["id", "orig_psnr", "orig_ssim", "psnr case 1", "psnr case 2", "psnr case 3", "psnr case 4", "ssim case 1", "ssim case 2", "ssim case 3", "ssim case 4"]
            for i in range(len(row_name)):
                writer.writerow([row_name[i]]+all_score[i])
        