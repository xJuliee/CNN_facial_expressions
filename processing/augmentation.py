import os
import numpy as np
from PIL import Image
import albumentations as A

DATA_SPLIT_PATH = '../data_split'  # Root directory
TARGET_CLASSES = ['confused']
AUGMENTED_SUFFIX = 'aug'

# Define minor grayscale augmentations WITHOUT rotation
transform = A.Compose([
    A.Affine(translate_percent={"x": 0.05, "y": 0.05}, p=0.5),  # shift only
    A.HorizontalFlip(p=0.5),  # horizontal mirror
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8,8), p=0.3),
])

def augment_and_save(image_path, output_dir, count):
    # Open image as grayscale
    image = Image.open(image_path).convert('L')
    image_np = np.array(image)

    # Albumentations expects HWC even for grayscale, so expand dims
    image_np_expanded = np.expand_dims(image_np, axis=2)
    augmented = transform(image=image_np_expanded)
    aug_img = augmented["image"].squeeze()  # Remove channel dim

    aug_pil = Image.fromarray(aug_img.astype(np.uint8))
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    new_filename = f"{base_name}_{AUGMENTED_SUFFIX}_{count}.png"
    save_path = os.path.join(output_dir, new_filename)
    aug_pil.save(save_path)
    print(f"Saved augmented image: {new_filename}")

def process_class_folder(base_dir, split, class_name, augmentations_per_image=3):
    class_path = os.path.join(base_dir, split, class_name)
    if not os.path.exists(class_path):
        print(f"[Skip] {class_path} does not exist.")
        return

    images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Augmenting {split}/{class_name} with {len(images)} images...")

    for i, img_name in enumerate(images):
        img_path = os.path.join(class_path, img_name)

        # The original image is kept intact

        # Create augmented versions per original image
        for aug_i in range(augmentations_per_image):
            try:
                augment_and_save(img_path, class_path, f"{i}_{aug_i}")
            except Exception as e:
                print(f"[Error] Could not augment {img_path}: {e}")

# Run for each split and class
for split in ['train', 'val', 'test']:
    for cls in TARGET_CLASSES:
        process_class_folder(DATA_SPLIT_PATH, split, cls)

print("✅ Augmentation complete.")