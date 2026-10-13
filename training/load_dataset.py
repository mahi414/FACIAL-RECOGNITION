import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
train_path = "dataset/train"

emotions = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]

X = []
y = []

for label, emotion in enumerate(emotions):

    folder = os.path.join(train_path, emotion)

    for filename in os.listdir(folder):

        image_path = os.path.join(folder, filename)

        image = Image.open(image_path).convert("L")

        image = np.array(image)

        X.append(image)
        y.append(label)

X = np.array(X)
y = np.array(y)

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Number of images:", len(X))
print("Number of labels:", len(y))
# Normalize pixel values
X = X / 255.0

print("Minimum pixel value:", X.min())
print("Maximum pixel value:", X.max())
X = X.reshape(-1, 48, 48, 1)

print("New X shape:", X.shape)


X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training images:", X_train.shape)
print("Validation images:", X_val.shape)
print("Training labels:", y_train.shape)
print("Validation labels:", y_val.shape)