import os
import shutil
import random

def split_data(source_dir, dest_dir, label="confused", train_ratio=0.72, val_ratio=0.18, test_ratio=0.10):
    # Create directories
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(dest_dir, split, label)
        os.makedirs(split_path, exist_ok=True)

    # Get all image files
    all_files = [f for f in os.listdir(source_dir)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    random.shuffle(all_files)

    total = len(all_files)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_files = all_files[:train_end]
    val_files = all_files[train_end:val_end]
    test_files = all_files[val_end:]

    # Function to copy files
    def copy_files(file_list, split_name):
        for filename in file_list:
            src = os.path.join(source_dir, filename)
            dst = os.path.join(dest_dir, split_name, label, filename)
            shutil.copy2(src, dst)

    copy_files(train_files, 'train')
    copy_files(val_files, 'val')
    copy_files(test_files, 'test')

    print(f"Total files: {total}")
    print(f"Train: {len(train_files)}")
    print(f"Validation: {len(val_files)}")
    print(f"Test: {len(test_files)}")
    print(f"Files copied into {dest_dir} with subfolders train/val/test and label folder '{label}'.")


# === USAGE ===
source_directory = "../new_confused"
destination_directory = "../data_split"

split_data(source_directory, destination_directory, label="confused")
