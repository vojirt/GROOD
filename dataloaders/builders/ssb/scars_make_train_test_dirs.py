import os

dataset_root = './_data/SemanticShiftBenchmark/StanfordCars'

class_names_file = os.path.join(dataset_root, 'class_names.txt')
data_file = os.path.join(dataset_root, 'data.txt')

class_names = []
with open(class_names_file) as f:
    lines = f.readlines()
    for line in lines:
        class_names.append(line.strip())

train_image_list = []
train_class_list = []
test_image_list = []
test_class_list = []
with open(data_file) as f:
    lines = f.readlines()

    for line in lines:
        fname, cls, istest = line.strip().split(",")
        cls = int(cls) - 1
        if int(istest) == 0:
            train_image_list.append(fname.strip())
            train_class_list.append(cls)
        else:
            test_image_list.append(fname.strip())
            test_class_list.append(cls)

# make links to the training images
for cls, img_name in zip(train_class_list, train_image_list):
    out_dir = os.path.join(dataset_root, 'train', class_names[cls])
    os.makedirs(out_dir, exist_ok=True)
    os.symlink(os.path.join(dataset_root, 'car_ims', img_name), os.path.join(out_dir, img_name))

# make links to the test images
for cls, img_name in zip(test_class_list, test_image_list):
    out_dir = os.path.join(dataset_root, 'test', class_names[cls])
    os.makedirs(out_dir, exist_ok=True)
    os.symlink(os.path.join(dataset_root, 'car_ims', img_name), os.path.join(out_dir, img_name))

