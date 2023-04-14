import os
import glob
import tqdm
import shutil


if __name__ == "__main__":
    root_dir = "<path_to>/M3SDA/"

    sub_datasets = [d for d in glob.glob(os.path.join(root_dir, "*")) if os.path.isdir(d)]
    for dataset in sub_datasets:
        print(f"Processing dataset {dataset} ...")
        dn = os.path.basename(dataset)
        with open(os.path.join(dataset, dn + "_train.txt"), "r") as f:
            train_list = f.readlines() 
        with open(os.path.join(dataset, dn + "_test.txt"), "r") as f:
            test_list = f.readlines() 

        train_dir = os.path.join(dataset, "train")
        test_dir = os.path.join(dataset, "test")

        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        for line in tqdm.tqdm(train_list):
            l = line.split(" ")[0]
            dirr = os.path.basename(os.path.dirname(l))
            file_name = os.path.basename(l) 
            os.makedirs(os.path.join(train_dir, dirr), exist_ok=True)
            shutil.copy2(os.path.join(dataset, dn, dirr, file_name), os.path.join(train_dir, dirr, file_name))
        for line in tqdm.tqdm(test_list):
            l = line.split(" ")[0]
            dirr = os.path.basename(os.path.dirname(l))
            file_name = os.path.basename(l) 
            os.makedirs(os.path.join(test_dir, dirr), exist_ok=True)
            shutil.copy2(os.path.join(dataset, dn, dirr, file_name), os.path.join(test_dir, dirr, file_name))


        



