import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

model = load_model("models/emotion_model.keras")

emotions = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]

image_path = "dataset/test/happy/"

import os

files = os.listdir(image_path)

image_path = os.path.join(image_path, files[0])

print("Testing:", image_path)

image = Image.open(image_path).convert("L")

image = np.array(image)

image = image / 255.0

image = image.reshape(1, 48, 48, 1)

prediction = model.predict(image, verbose=0)

index = np.argmax(prediction)

print("Predicted emotion:", emotions[index])
print("Confidence:", prediction[0][index] * 100, "%")