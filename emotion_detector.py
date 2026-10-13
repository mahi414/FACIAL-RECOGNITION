import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

# =========================
# Load emotion model
# =========================

model = tf.keras.models.load_model("emotion_model.keras")

emotion_labels = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]

# =========================
# MediaPipe face detector
# =========================

mp_face_detection = mp.solutions.face_detection

face_detection = mp_face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5
)

# =========================
# Webcam
# =========================

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Could not open webcam")
    exit()

print("Camera started. Press Q to quit.")

# =========================
# Main loop
# =========================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Could not read webcam")
        break

    # Convert BGR -> RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    results = face_detection.process(rgb_frame)

    if results.detections:

        for detection in results.detections:

            bbox = detection.location_data.relative_bounding_box

            h, w, _ = frame.shape

            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)

            box_w = int(bbox.width * w)
            box_h = int(bbox.height * h)

            # Keep coordinates inside image
            x = max(0, x)
            y = max(0, y)

            x2 = min(w, x + box_w)
            y2 = min(h, y + box_h)

            # Make sure bounding box is valid
            if x2 <= x or y2 <= y:
                continue

            # =========================
            # Extract face
            # =========================

            face = frame[y:y2, x:x2]

            if face.size == 0:
                continue

            # Convert face to grayscale
            face_gray = cv2.cvtColor(
                face,
                cv2.COLOR_BGR2GRAY
            )

            # Resize
            face_gray = cv2.resize(
                face_gray,
                (48, 48)
            )

            # Normalize
            face_gray = face_gray.astype(
                np.float32
            ) / 255.0

            # Shape -> (1,48,48,1)
            face_gray = face_gray.reshape(
                1, 48, 48, 1
            )

            # =========================
            # Predict emotion
            # =========================

            prediction = model(
                face_gray,
                training=False
            ).numpy()

            emotion_index = np.argmax(
                prediction[0]
            )

            emotion = emotion_labels[
                emotion_index
            ]

            confidence = (
                prediction[0][emotion_index] * 100
            )

            # =========================
            # Draw bounding box
            # =========================

            cv2.rectangle(
                frame,
                (x, y),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # =========================
            # Display emotion
            # =========================

            text = f"{emotion}: {confidence:.1f}%"

            cv2.putText(
                frame,
                text,
                (x, max(30, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

    # Show webcam
    cv2.imshow(
        "Facial Emotion Detection",
        frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# =========================
# Cleanup
# =========================

cap.release()
cv2.destroyAllWindows()
face_detection.close()