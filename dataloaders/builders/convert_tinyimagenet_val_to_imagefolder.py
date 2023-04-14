import os
import tqdm
import shutil


if __name__ == "__main__":
    root_dir = "./_data/tinyimagenet/tiny-imagenet-200/val_orig/"
    out_dir = "./_data/tinyimagenet/tiny-imagenet-200/val/"
    
    with open(os.path.join(root_dir, "val_annotations.txt"), "r") as fobj:
        lines = fobj.readlines() 

    for line in tqdm.tqdm(lines):
        ls = line.split()
        img_name = ls[0]
        img_class = ls[1]
        class_dir = os.path.join(out_dir, img_class)
        os.makedirs(class_dir, exist_ok=True)
        shutil.copy2(os.path.join(root_dir, "images", img_name), os.path.join(class_dir, img_name))
