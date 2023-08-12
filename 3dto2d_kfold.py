import os, sys, json, argparse
# import random
import numpy as np
import nibabel as nib

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

def make_dir_if_not_exist(dir):
    if not os.path.isdir(dir):
        os.mkdir(dir)
    else:
        print("Failed: Out File already exist, cannot make new directory.")
        sys.exit()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_name", "-P",help="select [EarlyFrame] or [LowDose]", type=str)
    parser.add_argument("--total_fold", help="total fold number", type=int, default=10)
    parser.add_argument("--fold", help="specify current fold [1~total_fold]", type=int, default=1)
    parser.add_argument("--data_dir", help="original nifti data directory", type=str, default="/home/kszuyen/project/PIBNii_Final")
    parser.add_argument("--json_file", help="json file for spliting into k fold", type=str)
    args = parser.parse_args()
    P = args.project_name
    K = args.total_fold
    FOLD = args.fold
    json_file = args.json_file

    data_dir = args.data_dir
    outfile = os.path.join(DIR_PATH, f"2d_data_{P}_fold{FOLD}")
    if P == "LowDose":
        MODALITIES = ["CT_RemoveTable_PET", "MR_PET", "PET", "PET_Coreg_Avg"] # Low Dose
    elif P == "EarlyFrame":
        MODALITIES = ["CT_RemoveTable_PET", "MR_PET", "Early_Frame", "PET_Coreg_Avg"] # Early Frame
    elif P == "LowDose_with_T1":
        MODALITIES = ["CT_RemoveTable_PET", "T1_PET", "PET", "PET_Coreg_Avg"] # Low Dose (MR changes from T2 to T1)
    GROUND_TRUTH = MODALITIES[3]
    print(f"Fold: {FOLD}")
    
    all_patient_files = [f for f in os.listdir(data_dir) if not f.startswith('.')] # 排除 “.DS_store“
    # patient_files = [f for f in all_patient_files if set(MODALITIES).issubset(os.listdir(os.path.join(data_dir, f)))] # 只保留包含所有需要資料的病人
    
    with open(os.path.join(DIR_PATH, json_file), 'r') as openfile:
        # Reading from json file
        json_object = json.load(openfile)
    # print(json_object)

    validation_index = (FOLD + K - 3) % K
    testing_index = (FOLD + K - 2) % K
    val_files = json_object[validation_index]
    test_files = json_object[testing_index]

    train_files = [patient for i, j in enumerate(json_object) if i not in (validation_index, testing_index) for patient in j]
    # print(len(train_files), len(val_files), len(test_files))
    assert len(set(train_files + val_files + test_files)) == len(all_patient_files)


    # total_patient_num = len(patient_files)

    # val_num = round(total_patient_num*0.1)
    # test_num = round(total_patient_num*0.1)
    # train_num = total_patient_num - val_num - test_num

    # print(f"total available patient num: {total_patient_num}")
    # print(f"split: {train_num}/{val_num}/{test_num}")
    # assert train_num + val_num + test_num == total_patient_num

    # train_files = random.sample(patient_files, train_num)

    # rest = [f for f in patient_files if f not in train_files]
    # val_files = random.sample(rest, val_num)
    # test_files = [f for f in rest if f not in val_files]

    print(f"Training set:\n{train_files}\n")
    print(f"Validation set:\n{val_files}\n")
    print(f"Testing set:\n{test_files}\n")

    print("Processing 3d to 2d...")
    make_dir_if_not_exist(outfile)
    for phase, patient_files in zip(["train", "val", "test"], [train_files, val_files, test_files]):
        make_dir_if_not_exist(os.path.join(outfile, phase))
        make_dir_if_not_exist(os.path.join(outfile, phase, "data"))
        make_dir_if_not_exist(os.path.join(outfile, phase, "ground_truth"))
        for patient in patient_files:
            image_dict = {}
            for modality in MODALITIES:
                # if modality == "PET":
                #     image_name = [f for f in sorted(os.listdir(os.path.join(data_dir, patient, modality))) if f.endswith('.nii') and f.startswith('s')][-3]
                # if "CT" in modality:
                #     image_name = [f for f in os.listdir(os.path.join(data_dir, patient, modality)) if f.endswith('.nii') and f.startswith('modified')][0]
                # else:
                image_name = [f for f in os.listdir(os.path.join(data_dir, patient, modality)) if f.endswith('.nii') and f.startswith('CGUN')][0]
                image_dict[modality] = nib.load(os.path.join(data_dir, patient, modality, image_name)).get_fdata()
            IMAGE_SHAPE = image_dict[MODALITIES[0]].shape
            TRAIN_CHANNELS = len(image_dict) - 1
            for z in range(IMAGE_SHAPE[2]):
                train_data = np.empty((IMAGE_SHAPE[0], IMAGE_SHAPE[1], TRAIN_CHANNELS))
                for i, modality in enumerate({m:image_dict[m] for m in image_dict if m!=GROUND_TRUTH}):
                    train_data[:, :, i] = image_dict[modality][:, :, z]
                if not np.isnan(train_data).any() and not np.isnan(image_dict[GROUND_TRUTH][:,:,z]).any():
                    np.save(os.path.join(outfile, phase, "data", f"{patient}_{z}.npy"), train_data)
                    np.save(os.path.join(outfile, phase, "ground_truth", f"{patient}_{z}.npy"), image_dict[GROUND_TRUTH][:,:,z])

    print("~~~Finished~~~")

    with open(os.path.join(outfile, 'train_val_test_split.txt'), 'w') as f:
        f.write(f"fold: {FOLD}")
        f.write(f"split: {len(train_files)}/{len(val_files)}/{len(test_files)}\n")
        f.write(f"Training set:\n{train_files}\n")
        f.write(f"Validation set:\n{val_files}\n")
        f.write(f"Testing set:\n{test_files}\n")



if __name__ == "__main__":
    main()




