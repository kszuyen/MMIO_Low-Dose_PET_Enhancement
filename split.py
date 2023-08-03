import os, argparse
import random
import json

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

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
    parser.add_argument("--total_fold", help="total fold number", type=int, default=10)
    parser.add_argument("--data_dir", help="original nifti data directory", type=str, default="/home/kszuyen/project/PIBNii_Final")
    parser.add_argument("--json_file", help="json file for spliting into k fold", type=str)
    args = parser.parse_args()
    data_dir = args.data_dir
    TOTAL_FOLD = args.total_fold
    json_file = args.json_file

    if os.path.exists(json_file):
        with open(os.path.join(DIR_PATH, json_file), 'r') as openfile:
        # Reading from json file
            json_object = json.load(openfile)
        assert len(json_object) == TOTAL_FOLD, \
            "Error: Json file already exists, and the fold number doesn't match."
        print("Json file already exists. Will not split patients again.")
    else:
        patient_files = [f for f in os.listdir(data_dir) if not f.startswith('.')] # 排除 “.DS_store“
        # patient_files = [f for f in all_patient_files if set(MODALITIES).issubset(os.listdir(os.path.join(data_dir, f)))] # 只保留包含所有需要資料的病人
        split = chunk(patient_files, TOTAL_FOLD)
        with open(os.path.join(DIR_PATH, json_file), "w") as outfile:
            json.dump(split, outfile, indent=4)

        print(len(patient_files))
        print(split)

if __name__ == "__main__":
    main()





