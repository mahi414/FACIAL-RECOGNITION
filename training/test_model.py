import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

emotions = [
    "angry", "disgust", "fear", "happy",
    "neutral", "sad", "surprise"
]

X_test, y_test = [], []
test_path = "dataset/test"

# Load and preprocess test images
for label, emotion in enumerate(emotions):
    folder = os.path.join(test_path, emotion)

    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Missing folder: {folder}")

    for filename in sorted(os.listdir(folder)):
        image_path = os.path.join(folder, filename)

        try:
            with Image.open(image_path) as img:
                image = np.array(
                    img.convert("L").resize((48, 48)),
                    dtype=np.float32
                )
            X_test.append(image)
            y_test.append(label)
        except (OSError, ValueError):
            print("Skipping unreadable image:", image_path)

X_test = np.array(X_test, dtype=np.float32)[..., np.newaxis] / 255.0
y_test = np.array(y_test)

print("Test images:", len(X_test))

for model_path in [
    "models/emotion_model.keras",
    "models/emotion_model_candidate.keras"
]:
    print("\nTesting:", model_path)

    if not os.path.exists(model_path):
        print("Model not found. Skipping.")
        continue

    model = load_model(model_path)
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

    print(f"Test loss: {loss:.4f}")
    print(f"Test accuracy: {accuracy * 100:.2f}%")
