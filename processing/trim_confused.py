import os
import cv2
import re

def preprocess_face_image(image_path, output_path, size=(75, 75)):
    image = cv2.imread(image_path)
    if image is None:
        print(f"[Warning] Cannot read image: {image_path}")
        return False

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
        cx, cy = x + w // 2, y + h // 2
        crop_size = int(max(w, h) * 1.2)

        x1 = max(0, cx - crop_size // 2)
        y1 = max(0, cy - crop_size // 2)
        x2 = min(image.shape[1], cx + crop_size // 2)
        y2 = min(image.shape[0], cy + crop_size // 2)

        cropped = image[y1:y2, x1:x2]
        resized = cv2.resize(cropped, size)
        gray_face = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        cv2.imwrite(output_path, gray_face)
        print(f"[Processed: Face] {os.path.basename(output_path)}")

        return True  # Important to return True here!
    else:
        # No face found: fallback to full image resize and grayscale
        resized = cv2.resize(image, size)
        gray_image = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(output_path, gray_image)
        print(f"[Processed: Fallback] {os.path.basename(output_path)}")

        return True

def get_last_index(output_dir, prefix):
    if not os.path.exists(output_dir):
        return None

    existing_files = os.listdir(output_dir)
    pattern = re.compile(rf"{re.escape(prefix)}(\d+)\.png")
    indices = []

    for f in existing_files:
        match = pattern.match(f)
        if match:
            indices.append(int(match.group(1)))

    return max(indices) if indices else None

def process_directory(input_dir, output_dir, prefix="confused"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    last_index = get_last_index(output_dir, prefix)
    if last_index is None:
        counter = 1  # start from 1 or change to 1497 if you want
    else:
        counter = last_index + 1

    image_files = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ])

    for filename in image_files:
        input_path = os.path.join(input_dir, filename)
        new_filename = f"{prefix}{counter}.png"
        output_path = os.path.join(output_dir, new_filename)

        success = preprocess_face_image(input_path, output_path)
        if success:
            os.remove(input_path)  # Remove original image
            counter += 1
        else:
            print(f"[Skipped] {filename}")

# === PATHS ===
input_directory = "confused"         # Folder containing raw images
output_directory = "new_confused"    # Folder to save processed & renamed images

# Run the batch processor
process_directory(input_directory, output_directory)

print("✅ All images processed and renamed.")