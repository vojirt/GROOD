import os

dataset_root = './_data/SemanticShiftBenchmark/FGVC-Aircraft/fgvc-aircraft-2013b/data'

trn_data_file = os.path.join(dataset_root, 'images_variant_trainval.txt')
tst_data_file = os.path.join(dataset_root, 'images_variant_test.txt')

train_image_list = []
train_class_list = []
with open(trn_data_file) as f:
    lines = f.readlines()
    for line in lines:
        fname, cls = line.strip().split(" ", maxsplit=1)
        train_image_list.append(fname.strip())
        train_class_list.append(cls.strip().replace('/', '-'))    # some variant names contanin a slash!

test_image_list = []
test_class_list = []
with open(tst_data_file) as f:
    lines = f.readlines()
    for line in lines:
        fname, cls = line.strip().split(" ", maxsplit=1)
        test_image_list.append(fname.strip())
        test_class_list.append(cls.strip().replace('/', '-'))    # some variant names contanin a slash!

# make links to the training images
for cls, img_name in zip(train_class_list, train_image_list):
    img_name = img_name + '.jpg'
    out_dir = os.path.join(dataset_root, 'train', cls)
    os.makedirs(out_dir, exist_ok=True)
    os.symlink(os.path.join(dataset_root, 'images', img_name), os.path.join(out_dir, img_name))

# make links to the test images
for cls, img_name in zip(test_class_list, test_image_list):
    img_name = img_name + '.jpg'
    out_dir = os.path.join(dataset_root, 'test', cls)
    os.makedirs(out_dir, exist_ok=True)
    os.symlink(os.path.join(dataset_root, 'images', img_name), os.path.join(out_dir, img_name))

