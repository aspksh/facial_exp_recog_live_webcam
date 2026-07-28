import av
import cv2
import torch
import numpy as np
import streamlit as st

from PIL import Image
from torchvision import transforms
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
st.write("Cascade Loaded:", not face_cascade.empty())
from model import EmotionCNN



# ---------------- Device ----------------

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

# ---------------- Model ----------------

model = EmotionCNN().to(device)

model.load_state_dict(
    torch.load(
        "facial_expression_recognition.pth",
        map_location=device
    )
)

model.eval()

EMOTIONS = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]

transform = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((48,48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ---------------- Face Detector ----------------


class EmotionProcessor(VideoProcessorBase):

   def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        # Test text
        cv2.putText(
            img,
            "HELLO",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        return av.VideoFrame.from_ndarray(img, format="bgr24")

import streamlit as st
import sys

st.write("Python:", sys.version)
st.write("OpenCV:", cv2.__version__)
st.write("Has CascadeClassifier:", hasattr(cv2, "CascadeClassifier"))

if hasattr(cv2, "CascadeClassifier"):
    st.write("CascadeClassifier exists")
else:
    st.write("CascadeClassifier NOT found")


st.set_page_config(
    page_title="Live Facial Expression Recognition",
    layout="centered"
)
st.title("Live Facial Expression Recognition")

webrtc_streamer(
key="emotion",
video_processor_factory=EmotionProcessor,
media_stream_constraints={
    "video": True,
    "audio": False
}
)

