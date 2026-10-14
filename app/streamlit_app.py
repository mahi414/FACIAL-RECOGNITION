
import streamlit as st
import numpy as np
import cv2
import json
import base64
import urllib.request
import threading

from PIL import Image, ImageDraw
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from av import VideoFrame


# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="Facial Emotion Detection",
    page_icon="😊",
    layout="wide"
)

st.title("😊 Facial Emotion Detection")
st.write(
    "Detect facial expressions using your webcam or an uploaded image."
)

EMOTION_SERVER = "http://127.0.0.1:8765"

EMOTIONS = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]

CASCADE_PATH = (
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)

face_detector = cv2.CascadeClassifier(CASCADE_PATH)
prediction_lock = threading.Lock()

if face_detector.empty():
    st.error("Could not load the face detector.")
    st.stop()


# ---------------- FACE DETECTION ----------------

def detect_faces(gray_image):
    return face_detector.detectMultiScale(
        gray_image,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(60, 60)
    )


# ---------------- EMOTION PREDICTION ----------------

def predict_emotion(face_crop):
    face_48 = cv2.resize(
        face_crop,
        (48, 48),
        interpolation=cv2.INTER_AREA
    )

    encoded = base64.b64encode(
        face_48.astype(np.uint8).tobytes()
    ).decode("ascii")

    payload = json.dumps({
        "image": encoded
    }).encode("utf-8")

    request = urllib.request.Request(
        EMOTION_SERVER,
        data=payload,
        headers={
            "Content-Type": "application/json"
        },
        method="POST"
    )

    with prediction_lock:
        with urllib.request.urlopen(
            request,
            timeout=10
        ) as response:
            result = json.loads(
                response.read().decode("utf-8")
            )

    if "error" in result:
        raise RuntimeError(result["error"])

    return result


# ---------------- DISPLAY ALL SCORES ----------------

def display_results(result):
    highest = result["highest_emotion"]
    confidence = result["confidence"]
    scores = result["emotions"]

    st.success(
        f"Highest Predicted Emotion: {highest} "
        f"({confidence:.2f}%)"
    )

    st.subheader("All 7 Emotion Scores")

    # Highest-scoring emotion first
    sorted_scores = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    for emotion, score in sorted_scores:
        st.write(f"**{emotion}: {score:.2f}%**")
        st.progress(
            max(0.0, min(float(score) / 100.0, 1.0))
        )

    st.caption(
        "Scores represent the model's outputs, not a "
        "certain measurement of a person's actual feelings."
    )


# ---------------- INPUT SELECTION ----------------

input_option = st.radio(
    "Choose your input method:",
    ["Live Webcam", "Upload Image"],
    horizontal=True
)


# ==================================================
# OPTION 1: LIVE WEBCAM
# ==================================================

if input_option == "Live Webcam":

    st.subheader("📹 Live Facial Emotion Detection")
    st.write(
        "Allow camera access and keep your face visible."
    )

    class EmotionProcessor(VideoProcessorBase):

        def __init__(self):
            self.emotion = "Looking for face..."
            self.confidence = 0.0
            self.scores = {}
            self.frame_count = 0
            self.last_error = None

        def recv(self, frame):
            image = frame.to_image().convert("RGB")
            rgb = np.asarray(image)
            gray = cv2.cvtColor(
                rgb,
                cv2.COLOR_RGB2GRAY
            )

            faces = detect_faces(gray)
            self.frame_count += 1

            draw = ImageDraw.Draw(image)

            if len(faces) > 0:
                # Select the largest detected face
                x, y, w, h = max(
                    faces,
                    key=lambda face: face[2] * face[3]
                )

                margin_x = int(w * 0.12)
                margin_y = int(h * 0.12)

                x1 = max(0, x - margin_x)
                y1 = max(0, y - margin_y)
                x2 = min(gray.shape[1], x + w + margin_x)
                y2 = min(gray.shape[0], y + h + margin_y)

                face_crop = gray[y1:y2, x1:x2]

                # Send every 10th frame for prediction
                if (
                    self.frame_count % 10 == 0
                    and face_crop.size > 0
                ):
                    try:
                        result = predict_emotion(face_crop)

                        self.emotion = result["highest_emotion"]
                        self.confidence = result["confidence"]
                        self.scores = result["emotions"]
                        self.last_error = None

                    except Exception as error:
                        self.emotion = "Server unavailable"
                        self.confidence = 0.0
                        self.scores = {}
                        self.last_error = str(error)
                        print("Prediction failed:", error)

                # Draw face rectangle
                draw.rectangle(
                    (x, y, x + w, y + h),
                    outline="lime",
                    width=3
                )

                # Show all scores on the video
                panel_width = 260
                panel_height = 38 + (
                    len(self.scores) * 22
                )

                draw.rectangle(
                    (10, 10, panel_width, panel_height),
                    fill="black"
                )

                draw.text(
                    (18, 15),
                    f"Highest: {self.emotion}",
                    fill="lime"
                )

                if self.scores:
                    sorted_scores = sorted(
                        self.scores.items(),
                        key=lambda item: item[1],
                        reverse=True
                    )

                    for i, (name, score) in enumerate(
                        sorted_scores
                    ):
                        draw.text(
                            (18, 38 + i * 22),
                            f"{name}: {score:.1f}%",
                            fill="white"
                        )
                else:
                    draw.text(
                        (18, 40),
                        "Waiting for prediction...",
                        fill="white"
                    )

            else:
                self.emotion = "No face detected"
                self.confidence = 0.0
                self.scores = {}

                draw.rectangle(
                    (10, 10, 230, 42),
                    fill="black"
                )
                draw.text(
                    (18, 17),
                    "No face detected",
                    fill="white"
                )

            return VideoFrame.from_image(image)

    webrtc_streamer(
        key="emotion-webcam",
        video_processor_factory=EmotionProcessor,
        media_stream_constraints={
            "video": True,
            "audio": False
        },
        async_processing=True
    )

    st.caption(
        "The largest detected face is classified. "
        "Scores update periodically while the camera is running."
    )


# ==================================================
# OPTION 2: UPLOAD IMAGE
# ==================================================

else:

    st.subheader("🖼️ Upload an Image")
    st.write(
        "Upload a JPG, JPEG, or PNG image containing a visible face."
    )

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        try:
            original_image = Image.open(
                uploaded_file
            ).convert("RGB")

            image_array = np.array(original_image)

            gray_image = cv2.cvtColor(
                image_array,
                cv2.COLOR_RGB2GRAY
            )

            st.image(
                original_image,
                caption="Uploaded Image",
                use_container_width=True
            )

            if st.button(
                "Detect Emotion",
                type="primary"
            ):

                faces = detect_faces(gray_image)

                if len(faces) == 0:
                    st.warning(
                        "No face detected. Try a clear, "
                        "front-facing image with good lighting."
                    )

                else:
                    # Select the largest face
                    x, y, w, h = max(
                        faces,
                        key=lambda face: face[2] * face[3]
                    )

                    margin_x = int(w * 0.12)
                    margin_y = int(h * 0.12)

                    x1 = max(0, x - margin_x)
                    y1 = max(0, y - margin_y)
                    x2 = min(
                        gray_image.shape[1],
                        x + w + margin_x
                    )
                    y2 = min(
                        gray_image.shape[0],
                        y + h + margin_y
                    )

                    face_crop = gray_image[y1:y2, x1:x2]

                    with st.spinner(
                        "Analyzing facial expression..."
                    ):
                        result = predict_emotion(face_crop)

                    # Draw a box around the detected face
                    result_image = original_image.copy()
                    draw = ImageDraw.Draw(result_image)

                    draw.rectangle(
                        (x, y, x + w, y + h),
                        outline="lime",
                        width=4
                    )

                    st.subheader("Detected Face")
                    st.image(
                        result_image,
                        use_container_width=True
                    )

                    # Display highest emotion and all scores
                    display_results(result)

        except Exception as error:
            st.error(
                "Could not process the image or contact "
                "the prediction server."
            )
            st.caption(str(error))

