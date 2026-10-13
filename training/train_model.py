import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import layers, models


# -------------------------
# 1. Load Dataset
# -------------------------

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


# -------------------------
# 2. Preprocessing
# -------------------------

X = X / 255.0

X = X.reshape(-1, 48, 48, 1)


# -------------------------
# 3. Train / Validation Split
# -------------------------

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training images:", X_train.shape)
print("Validation images:", X_val.shape)


# -------------------------
# 4. Build CNN Model
# -------------------------

model = models.Sequential([

    layers.Input(shape=(48, 48, 1)),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),

    layers.Dense(128, activation="relu"),

    layers.Dropout(0.5),

    layers.Dense(7, activation="softmax")
])


# -------------------------
# 5. Compile Model
# -------------------------

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# -------------------------
# 6. Train Model
# -------------------------

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=64
)


# -------------------------
# 7. Save Model
# -------------------------

os.makedirs("models", exist_ok=True)

model.save("models/emotion_model.keras")

print("Model saved successfully!")
