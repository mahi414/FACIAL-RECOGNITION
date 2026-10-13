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

st.set_page_config(
    page_title="Facial Emotion Detection",
    page_icon="😊",
    layout="wide"
)

st.title("😊 Facial Emotion Detection")
st.write("Detect facial emotions using a live webcam or an uploaded image.")

EMOTION_SERVER = "http://127.0.0.1:8765"

CASCADE_PATH = (
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)

face_detector = cv2.CascadeClassifier(CASCADE_PATH)
prediction_lock = threading.Lock()

if face_detector.empty():
    st.error("Could not load the face detector.")
    st.stop()


# Send a detected face to the separate TensorFlow server
def predict_emotion(face_crop):
    face_48 = cv2.resize(
        face_crop,
        (48, 48),
        interpolation=cv2.INTER_AREA
    )

    encoded = base64.b64encode(
        face_48.astype(np.uint8).tobytes()
    ).decode("ascii")

    payload = json.dumps({"image": encoded}).encode("utf-8")

    request = urllib.request.Request(
        EMOTION_SERVER,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    with prediction_lock:
        with urllib.request.urlopen(request, timeout=10) as response:
            result = json.loads(response.read().decode("utf-8"))

    if "error" in result:
        raise RuntimeError(result["error"])

    return result["emotion"], result["confidence"]


# Detect faces in an image
def detect_faces(gray_image):
    return face_detector.detectMultiScale(
        gray_image,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(60, 60)
    )


# Choose the input method
input_option = st.radio(
    "Choose your input method:",
    ["Live Webcam", "Upload Image"],
    horizontal=True
)


# OPTION 1: LIVE WEBCAM
if input_option == "Live Webcam":

    st.subheader("📹 Live Emotion Detection")
    st.write("Allow camera access and keep your face visible.")

    class EmotionProcessor(VideoProcessorBase):

        def __init__(self):
            self.emotion = "Looking for face..."
            self.confidence = 0.0
            self.frame_count = 0
            self.last_prediction_frame = -10

        def recv(self, frame):
            image = frame.to_image().convert("RGB")
            rgb = np.asarray(image)
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

            faces = detect_faces(gray)
            self.frame_count += 1

            if len(faces) > 0:
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

                # Predict every 10 frames
                if (
                    self.frame_count % 10 == 0
                    and face_crop.size > 0
                ):
                    try:
                        emotion, confidence = predict_emotion(
                            face_crop
                        )
                        self.emotion = emotion
                        self.confidence = confidence
                        self.last_prediction_frame = self.frame_count

                    except Exception as error:
                        self.emotion = "Server unavailable"
                        self.confidence = 0.0
                        print("Prediction failed:", error)

                draw = ImageDraw.Draw(image)
                draw.rectangle(
                    (x, y, x + w, y + h),
                    outline="lime",
                    width=3
                )
                draw.rectangle((x, max(0, y - 30), x + 240, y),
                               fill="black")
                draw.text(
                    (x + 5, max(0, y - 25)),
                    f"{self.emotion} ({self.confidence:.1f}%)",
                    fill="white"
                )

            else:
                self.emotion = "No face detected"
                self.confidence = 0.0

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
        "The green box marks the detected face. "
        "The largest detected face is classified."
    )


# OPTION 2: UPLOAD AN IMAGE
else:

    st.subheader("🖼️ Image Emotion Detection")
    st.write("Upload a clear photo containing a face.")

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        original_image = Image.open(uploaded_file).convert("RGB")
        image_array = np.array(original_image)
        gray_image = cv2.cvtColor(
            image_array,
            cv2.COLOR_RGB2GRAY
        )

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                original_image,
                caption="Uploaded Image",
                use_container_width=True
            )

        if st.button("Detect Emotion", type="primary"):

            faces = detect_faces(gray_image)

            if len(faces) == 0:
                st.warning(
                    "No face detected. Try a clearer, front-facing photo."
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

                try:
                    with st.spinner("Predicting emotion..."):
                        emotion, confidence = predict_emotion(
                            face_crop
                        )

                    # Draw face box on uploaded image
                    result_image = original_image.copy()
                    draw = ImageDraw.Draw(result_image)

                    draw.rectangle(
                        (x, y, x + w, y + h),
                        outline="lime",
                        width=4
                    )

                    with col2:
                        st.image(
                            result_image,
                            caption="Detected Face",
                            use_container_width=True
                        )

                    st.success(f"Predicted Emotion: {emotion}")
                    st.metric(
                        "Model Confidence",
                        f"{confidence:.2f}%"
                    )

                except Exception as error:
                    st.error(
                        "Prediction server unavailable. "
                        "Start prediction_server.py in another terminal."
                    )
                    st.caption(str(error))
