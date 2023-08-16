import os, argparse
import random
import json
import os, sys, json, argparse
import numpy as np
import nibabel as nib
from utils import *

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

def _3dto2d(project_name, data_dir, split, output_dir):

    # output_dir = os.path.join(DIR_PATH, f"{project_name}_2d_data")
    if project_name == "LowDose":
        MODALITIES = ["CT_RemoveTable_PET", "MR_PET", "PET", "PET_Coreg_Avg"] # Low Dose
    elif project_name == "EarlyFrame":
        MODALITIES = ["CT_RemoveTable_PET", "MR_PET", "Early_Frame", "PET_Coreg_Avg"] # Early Frame
    elif project_name == "LowDose_with_T1":
        MODALITIES = ["CT_RemoveTable_PET", "T1_PET", "PET", "PET_Coreg_Avg"] # Low Dose (MR changes from T2 to T1)
    GROUND_TRUTH = MODALITIES[3]

    print("Processing 3d to 2d...")
    make_dir(output_dir)
    make_dir(os.path.join(output_dir, "image_info"))
    image_info = dict()
    for f, patient_files in enumerate(split):
        make_dir(os.path.join(output_dir, f"group{f}"))
        make_dir(os.path.join(output_dir, f"group{f}", "data"))
        make_dir(os.path.join(output_dir, f"group{f}", "ground_truth"))
        for patient in patient_files:
            image_dict = {}
            for modality in MODALITIES:
                image_name = [f for f in os.listdir(os.path.join(data_dir, patient, modality)) if f.endswith('.nii') and f.startswith('CGUN')][0]
                data = nib.load(os.path.join(data_dir, patient, modality, image_name))
                image_dict[modality] = data.get_fdata()

                if patient not in image_info:
                    image_info[patient] = data.affine

            IMAGE_SHAPE = image_dict[MODALITIES[0]].shape
            TRAIN_CHANNELS = len(image_dict) - 1
            for z in range(IMAGE_SHAPE[2]):
                train_data = np.empty((IMAGE_SHAPE[0], IMAGE_SHAPE[1], TRAIN_CHANNELS))
                for i, modality in enumerate({m:image_dict[m] for m in image_dict if m!=GROUND_TRUTH}):
                    train_data[:, :, i] = image_dict[modality][:, :, z]
                if not np.isnan(train_data).any() and not np.isnan(image_dict[GROUND_TRUTH][:,:,z]).any():
                    np.save(os.path.join(output_dir, f"group{f}", "data", f"{patient}_{z}.npy"), train_data)
                    np.save(os.path.join(output_dir, f"group{f}", "ground_truth", f"{patient}_{z}.npy"), image_dict[GROUND_TRUTH][:,:,z])

    for patient in image_info:
        np.save(os.path.join(output_dir, "image_info", f"{patient}.npy"), np.array(image_info[patient]))
    with open(os.path.join(DIR_PATH, output_dir, "split.json"), "w") as output_dir:
        json.dump(split, output_dir, indent=4)

def chunk(xs, n):
    ys = list(xs)
    random.shuffle(ys)
    ylen = len(ys)
    size = int(ylen / n)
    chunks = [ys[0+size*i : size*(i+1)] for i in range(n)]
    leftover = ylen - size*n
    edge = size*n
    for i in range(leftover):
        chunks[i%n].append(ys[edge+i])

    return chunks

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_name", "-P",help="select [EarlyFrame] or [LowDose]", type=str)
    parser.add_argument("--data_dir", help="original 3d nifti data directory", type=str, default="/home/kszuyen/MMIO_Low-Dose_PET_Enhancement/PIBNii_MPR_T1")
    parser.add_argument("--total_fold", type=int, default=10)
    parser.add_argument("--split_json_file", type=str)
    args = parser.parse_args()
    project_name = args.project_name
    data_dir = args.data_dir
    TOTAL_FOLD = args.total_fold
    output_dir = os.path.join(DIR_PATH, f"{project_name}_2d_data")
    json_file = args.split_json_file

    if os.path.exists(output_dir):
        print(f"Already split: {output_dir}")
    else:
        if json_file:
            with open(json_file, 'r') as openfile:
                split = json.load(openfile)
        else:
            patient_files = [f for f in os.listdir(data_dir) if not f.startswith('.')] # 排除 “.DS_store“
            # patient_files = [f for f in all_patient_files if set(MODALITIES).issubset(os.listdir(os.path.join(data_dir, f)))] # 只保留包含所有需要資料的病人
            split = chunk(patient_files, TOTAL_FOLD)
        _3dto2d(project_name, data_dir, split, output_dir)
        print("---Finished---")

if __name__ == "__main__":
    main()





