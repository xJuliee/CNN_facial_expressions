import os

DATA_SPLIT_PATH = '../data_split'  # Root directory where train/val/test folders are


def remove_augmented_images(base_dir):
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(base_dir, split)
        if not os.path.exists(split_dir):
            print(f"[Skip] {split_dir} does not exist.")
            continue

        for class_name in os.listdir(split_dir):
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            for filename in os.listdir(class_dir):
                if '_aug' in filename and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    file_path = os.path.join(class_dir, filename)
                    try:
                        os.remove(file_path)
                        print(f"Deleted augmented image: {file_path}")
                    except Exception as e:
                        print(f"[Error] Could not delete {file_path}: {e}")


remove_augmented_images(DATA_SPLIT_PATH)
print("✅ Finished removing augmented images.")
