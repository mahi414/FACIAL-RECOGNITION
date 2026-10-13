import streamlit as st
import numpy as np
from PIL import ImageDraw
from tensorflow.keras.models import load_model
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase


st.set_page_config(page_title="Real-Time Emotion Detection")
st.title("Real-Time Facial Emotion Detection")
st.write("Allow webcam access and keep your face centered in the camera.")

emotions = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]


@st.cache_resource
def load_emotion_model():
    return load_model("models/emotion_model.keras")


model = load_emotion_model()


class EmotionProcessor(VideoProcessorBase):

    def __init__(self):
        self.emotion = "Detecting..."
        self.confidence = 0.0
        self.frame_count = 0

    def recv(self, frame):
        image = frame.to_image().convert("RGB")

        width, height = image.size

        # Crop a square from the center of the camera frame.
        # Keep your face centered for better predictions.
        side = min(width, height)
        left = (width - side) // 2
        top = (height - side) // 2

        face_crop = image.crop((
            left,
            top,
            left + side,
            top + side
        ))

        # Preprocess for the trained CNN
        gray = face_crop.convert("L").resize((48, 48))

        array = np.array(gray, dtype=np.float32) / 255.0
        array = array.reshape(1, 48, 48, 1)

        # Predict every third frame to reduce processing load
        self.frame_count += 1

        if self.frame_count % 3 == 0:
            prediction = model(array, training=False).numpy()[0]

            index = int(np.argmax(prediction))
            self.emotion = emotions[index]
            self.confidence = float(prediction[index] * 100)

        # Display prediction on the live video without OpenCV
        draw = ImageDraw.Draw(image)
        draw.rectangle((10, 10, 330, 55), fill="black")
        draw.text(
            (20, 20),
            f"{self.emotion} ({self.confidence:.1f}%)",
            fill="white"
        )

        from av import VideoFrame
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
    "Tip: Center your face in the frame. "
    "The model predicts emotions from the center crop; "
    "it does not independently detect face boundaries."
)
