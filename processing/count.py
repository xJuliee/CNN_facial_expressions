import os

def count_images_in_folders(base_path="data_split"):
    valid_extensions = ('.jpg', '.jpeg', '.png')
    folder_counts = {}

    for root, dirs, files in os.walk(base_path):
        for folder in dirs:
            folder_path = os.path.join(root, folder)
            count = sum(
                1 for f in os.listdir(folder_path)
                if f.lower().endswith(valid_extensions)
            )
            relative_path = os.path.relpath(folder_path, base_path)
            folder_counts[relative_path] = count

    return folder_counts

if __name__ == "__main__":
    image_counts = count_images_in_folders("../data_split")
    print("Image counts per class folder:\n")
    for folder, count in sorted(image_counts.items()):
        print(f"{folder}: {count} images")
