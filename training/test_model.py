import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("models/emotion_model.keras")

emotions = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]

test_path = "dataset/test"

X_test = []
y_test = []

# Load test images
for label, emotion in enumerate(emotions):

    folder = os.path.join(test_path, emotion)

    for filename in os.listdir(folder):

        image_path = os.path.join(folder, filename)

        image = Image.open(image_path).convert("L")

        image = np.array(image)

        X_test.append(image)
        y_test.append(label)

# Convert to NumPy arrays
X_test = np.array(X_test)
y_test = np.array(y_test)

# Normalize
X_test = X_test / 255.0

# Add grayscale channel
X_test = X_test.reshape(-1, 48, 48, 1)

print("Test images:", X_test.shape)
print("Test labels:", y_test.shape)

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)

print("Test Loss:", loss)
print("Test Accuracy:", accuracy)