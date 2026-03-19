import cv2
import random
import os

input_folder = "dataset/index"
output_folder = "dataset_aug/index"

os.makedirs(output_folder, exist_ok=True)

def augment_image(img):
    augmented = []

    angle = random.randint(-15, 15)

    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
    rotated = cv2.warpAffine(img, M, (w, h))

    augmented.append(rotated)

    # Brightness
    bright = cv2.convertScaleAbs(img, alpha=1, beta=50)
    dark = cv2.convertScaleAbs(img, alpha=1, beta=-50)
    augmented.extend([bright, dark])

    # Blur
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    augmented.append(blurred)

    return augmented

count = 0

for filename in os.listdir(input_folder):
    path = os.path.join(input_folder, filename)
    img = cv2.imread(path)

    if img is None:
        continue

    cv2.imwrite(os.path.join(output_folder, f"orig_{count}.jpg"), img)

    augmented_images = augment_image(img)

    for aug in augmented_images:
        cv2.imwrite(os.path.join(output_folder, f"aug_{count}.jpg"), aug)
        count += 1