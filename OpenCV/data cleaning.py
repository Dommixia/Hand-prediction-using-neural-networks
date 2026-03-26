from PIL import Image
import os

base_path = "dataset_aug"

for root, _, files in os.walk(base_path):
    for file in files:
        path = os.path.join(root, file)
        try:
            img = Image.open(path)
            img.verify()
        except:
            print("Deleting corrupted:", path)
            os.remove(path)