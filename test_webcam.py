import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from av import VideoFrame

st.title("Webcam Test — No TensorFlow")

class CameraProcessor(VideoProcessorBase):
    def recv(self, frame):
        return frame

webrtc_streamer(
    key="camera-test",
    video_processor_factory=CameraProcessor,
    media_stream_constraints={"video": True, "audio": False},
)

st.write("If the camera works, the video should appear above.")
