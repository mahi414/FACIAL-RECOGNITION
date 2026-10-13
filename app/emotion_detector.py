import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import cv2
import numpy as np
from tensorflow.keras.models import load_model


# Load model
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

# Face detector
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera could not be opened")
    exit()

print("Camera started. Press Q to quit.")

frame_count = 0
current_emotion = "Detecting..."
current_confidence = 0


while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    frame_count += 1

    for (x, y, w, h) in faces:

        # Only predict every 5th frame
        if frame_count % 5 == 0:

            face = gray[y:y+h, x:x+w]

            face = cv2.resize(face, (48, 48))

            face = face.astype("float32") / 255.0

            face = face.reshape(1, 48, 48, 1)

            # Prediction
            prediction = model(face, training=False).numpy()

            emotion_index = np.argmax(prediction)

            current_emotion = emotions[emotion_index]

            current_confidence = prediction[0][emotion_index] * 100

        # Draw rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            2
        )

        # Display result
        text = f"{current_emotion} {current_confidence:.1f}%"

        cv2.putText(
            frame,
            text,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2
        )

    cv2.imshow(
        "Facial Emotion Detection",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()