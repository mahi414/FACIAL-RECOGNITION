import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from av import VideoFrame
from tensorflow.keras.models import load_model

st.title("Webcam + TensorFlow Compatibility Test")

@st.cache_resource
def get_model():
    return load_model("models/emotion_model.keras")

model = get_model()
st.success("TensorFlow model loaded successfully!")

class CameraProcessor(VideoProcessorBase):
    def recv(self, frame):
        return frame

webrtc_streamer(
    key="camera-tf-test",
    video_processor_factory=CameraProcessor,
    media_stream_constraints={"video": True, "audio": False},
)

st.write("This test loads TensorFlow but does not run predictions.")
