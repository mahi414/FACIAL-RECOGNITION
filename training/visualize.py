import os
import matplotlib.pyplot as plt
from PIL import Image

train_path = "dataset/train"

emotions = os.listdir(train_path)

plt.figure(figsize=(12, 8))

for i, emotion in enumerate(emotions):
    folder = os.path.join(train_path, emotion)
    images = os.listdir(folder)

    image_path = os.path.join(folder, images[0])
    image = Image.open(image_path)

    plt.subplot(2, 4, i + 1)
    plt.imshow(image, cmap="gray")
    plt.title(emotion)
    plt.axis("off")

plt.tight_layout()

# Save instead of displaying
plt.savefig("training/emotion_samples.png")

print("Visualization saved successfully!")