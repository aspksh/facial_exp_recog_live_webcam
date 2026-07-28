import av
import cv2
import torch
import numpy as np


from PIL import Image
from torchvision import transforms
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

#face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

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

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(80,80)
        )

        for (x,y,w,h) in faces:

            # thoda margin add karte hain
            pad = 15

            x1 = max(0, x-pad)
            y1 = max(0, y-pad)

            x2 = min(img.shape[1], x+w+pad)
            y2 = min(img.shape[0], y+h+pad)

            face = img[y1:y2, x1:x2]

            if face.size == 0:
                continue

            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

            face = Image.fromarray(face)

            tensor = transform(face).unsqueeze(0).to(device)

            with torch.no_grad():

                output = model(tensor)

                probs = torch.softmax(output, dim=1)

                confidence, pred = torch.max(probs, dim=1)

            emotion = EMOTIONS[pred.item()]
            confidence = confidence.item()*100

            cv2.rectangle(
                img,
                (x1,y1),
                (x2,y2),
                (0,255,0),
                2
            )

            cv2.putText(
                img,
                f"{emotion} {confidence:.1f}%",
                (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0,255,0),
                2
            )

        return av.VideoFrame.from_ndarray(img, format="bgr24")

import streamlit as st
st.write(cv2)
st.write(cv2.__file__)
st.write(cv2.__version__)
st.write(hasattr(cv2, "CascadeClassifier"))


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

