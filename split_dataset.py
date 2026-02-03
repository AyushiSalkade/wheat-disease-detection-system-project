import os
import shutil
import random

original_dataset = "data/archive/wheat disease dataset"
base_dir = "dataset"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")

for cls in os.listdir(original_dataset):
    os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
    os.makedirs(os.path.join(val_dir, cls), exist_ok=True)

split_ratio = 0.8

for cls in os.listdir(original_dataset):
    cls_path = os.path.join(original_dataset, cls)
    images = os.listdir(cls_path)
    random.shuffle(images)
    
    split_index = int(len(images) * split_ratio)
    train_images = images[:split_index]
    val_images = images[split_index:]
    
    for img in train_images:
        shutil.copyfile(os.path.join(cls_path, img), os.path.join(train_dir, cls, img))
    for img in val_images:
        shutil.copyfile(os.path.join(cls_path, img), os.path.join(val_dir, cls, img))

print("Dataset split completed!")