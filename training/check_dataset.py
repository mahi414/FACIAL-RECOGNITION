import os
from PIL import Image

train_path = "dataset/train"

emotions = os.listdir(train_path)

print("Emotions:", emotions)

for emotion in emotions:
    folder = os.path.join(train_path, emotion)

    images = os.listdir(folder)

    print(f"{emotion}: {len(images)} images")

    if images:
        image_path = os.path.join(folder, images[0])

        image = Image.open(image_path)

        print("   Size:", image.size)
        print("   Mode:", image.mode)