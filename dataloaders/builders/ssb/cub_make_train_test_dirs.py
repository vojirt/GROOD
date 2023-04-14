import os

dataset_root = './_data/SemanticShiftBenchmark/cub-200-2011/CUB_200_2011'

image_list_file = os.path.join(dataset_root, 'images.txt')
split_file = os.path.join(dataset_root, 'train_test_split.txt')

image_list = []
with open(image_list_file) as f:
    lines = f.readlines()

    for line in lines:
        img_name = line.strip().split(" ")[1]
        image_list.append(img_name.strip())

# print(image_list)
# print(f"Image list length: {len(image_list)}")

train_image_list = []
test_image_list = []
with open(split_file) as f:
    lines = f.readlines()

    for line in lines:
        idx, is_training = line.strip().split(" ")
        idx = int(idx) - 1
        if int(is_training) == 1:
            train_image_list.append(image_list[idx])
        else:
            test_image_list.append(image_list[idx])

# print(train_image_list)

# make links to the training images
out_dir = os.path.join(dataset_root, 'train_images')
for img_name in train_image_list:
    dir, img_file_name = img_name.split("/")
    out_dir = os.path.join(dataset_root, 'train_images', dir)
    os.makedirs(out_dir, exist_ok=True)
    os.symlink(os.path.join(dataset_root, 'images', img_name), os.path.join(out_dir, img_file_name))

# make links to the test images
out_dir = os.path.join(dataset_root, 'test_images')
for img_name in test_image_list:
    dir, img_file_name = img_name.split("/")
    out_dir = os.path.join(dataset_root, 'test_images', dir)
    os.makedirs(out_dir, exist_ok=True)
    os.symlink(os.path.join(dataset_root, 'images', img_name), os.path.join(out_dir, img_file_name))
