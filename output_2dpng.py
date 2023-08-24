import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os, argparse, sys, json
from utils import *
from unet import UNET
from dataset import NTUH_dataset
from tqdm import tqdm
import matplotlib.pyplot as plt

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

def load_config(project_name, project_results_dir, fold, case, batch_size):
    cfg = dotdict(dict())
    cfg.project_name = project_name
    cfg.project_results_dir = project_results_dir
    cfg.fold = fold
    cfg.data_dir = os.path.join(DIR_PATH, f"{project_name}_2d_data")
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
        2: PT & CT
        3: PT & MR
        4: PT & CT & MR
        """)
        sys.exit()

    models_dir = os.path.join(project_results_dir, f"fold{fold}", "models_file")
    cfg.best_ckpt_dir = os.path.join(models_dir, f"{cfg.mod_name}_best.pth")

    return cfg

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument("--project_name", "-P",help="select [EarlyFrame] or [LowDose]", type=str)
    parser.add_argument("--project_results_dir", type=str)
    parser.add_argument("--max_z_slice", type=int, default=91)

    args = parser.parse_args()
    project_name = args.project_name
    project_results_dir = args.project_results_dir if args.project_results_dir else os.path.join(DIR_PATH, "results", project_name)
    json_file = os.path.join(DIR_PATH, f"{project_name}_2d_data", "split.json")
    max_z_slice = args.max_z_slice

    ct_pad_value, mr_pad_value, pt_pad_value = 0., 0., 0.
    output_dir = os.path.join(DIR_PATH, f'{project_name}_output_2d_png')
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

                all_mod_3dnparr_dict[patient] = dict()
                for mod in ['ct', 'mr', 'pt', 'gt', 'case1', 'case2', 'case3', 'case4']:
                    all_mod_3dnparr_dict[patient][mod] = dict()

            test_dataset = NTUH_dataset(
                root_dir=os.path.join(DIR_PATH, f"{project_name}_2d_data"),
                dataset_type="test",
                case=4,
                fold=fold,
                min_max_scaler=None,
            )

            for idx, (data, gt, patient, slc) in enumerate(test_dataset):
                all_mod_3dnparr_dict[patient]['ct'][int(slc)] = data[0].detach().cpu().numpy()
                all_mod_3dnparr_dict[patient]['mr'][int(slc)] = data[1].detach().cpu().numpy()
                all_mod_3dnparr_dict[patient]['pt'][int(slc)] = data[2].detach().cpu().numpy()
                all_mod_3dnparr_dict[patient]['gt'][int(slc)] = gt[0].detach().cpu().numpy()
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
                        fold=cfg.fold,
                        min_max_scaler=min_max_scaler,
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

                        for i, (p, s) in enumerate(zip(list(batch_patient), list(batch_slc))):
                            all_mod_3dnparr_dict[p][f"case{case}"] [int(s)] = reverse_normalize(scaler=min_max_scaler[2])(output[i][0]).detach().cpu().numpy()

            for patient in all_mod_3dnparr_dict:
                for z in all_mod_3dnparr_dict[patient]['ct']:
                    fig = plt.figure()
                    fig.add_subplot(2, 4, 1)
                    plt.imshow(all_mod_3dnparr_dict[patient]['ct'][z], cmap="gray")
                    plt.title("CT")
                    plt.axis('off')
                    fig.add_subplot(2, 4, 2)
                    plt.imshow(all_mod_3dnparr_dict[patient]['mr'][z], cmap="gray")
                    plt.title("MR")
                    plt.axis('off')
                    fig.add_subplot(2, 4, 3)
                    plt.imshow(all_mod_3dnparr_dict[patient]['pt'][z])
                    plt.title("PT")
                    plt.axis('off')
                    fig.add_subplot(2, 4, 4)
                    plt.imshow(all_mod_3dnparr_dict[patient]['gt'][z])
                    plt.title("GT")
                    plt.axis('off')
                    if all_mod_3dnparr_dict[patient]['case1']:
                        fig.add_subplot(2, 4, 5)
                        plt.imshow(all_mod_3dnparr_dict[patient]['case1'][z])
                        plt.title("Case1")
                        plt.axis('off')
                    if all_mod_3dnparr_dict[patient]['case2']:
                        fig.add_subplot(2, 4, 6)
                        plt.imshow(all_mod_3dnparr_dict[patient]['case2'][z])
                        plt.title("Case2")
                        plt.axis('off')
                    if all_mod_3dnparr_dict[patient]['case3']:
                        fig.add_subplot(2, 4, 7)
                        plt.imshow(all_mod_3dnparr_dict[patient]['case3'][z])
                        plt.title("Case3")
                        plt.axis('off')
                    if all_mod_3dnparr_dict[patient]['case4']:
                        fig.add_subplot(2, 4, 8)
                        plt.imshow(all_mod_3dnparr_dict[patient]['case4'][z])
                        plt.title("Case4")
                        plt.axis('off')

                    fig.suptitle(f"{patient}_{z}")
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, patient, f"{z}.png"))
                    plt.close()
