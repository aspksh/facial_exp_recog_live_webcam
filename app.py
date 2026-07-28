import av
import cv2
import torch
import numpy as np
import mediapipe as mp

from PIL import Image
from torchvision import transforms
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

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

mp_face = mp.solutions.face_detection


class EmotionProcessor(VideoProcessorBase):

    def __init__(self):

        self.face_detector = mp_face.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.5
        )

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = self.face_detector.process(rgb)

        if results.detections:

            h, w, _ = img.shape

            for detection in results.detections:

                box = detection.location_data.relative_bounding_box

                x = int(box.xmin * w)
                y = int(box.ymin * h)

                bw = int(box.width * w)
                bh = int(box.height * h)

                x = max(0, x)
                y = max(0, y)

                face = rgb[y:y+bh, x:x+bw]

                if face.size == 0:
                    continue

                face = Image.fromarray(face)

                tensor = transform(face).unsqueeze(0).to(device)

                with torch.no_grad():

                    output = model(tensor)

                    probs = torch.softmax(output,1)

                    conf, pred = torch.max(probs,1)

                emotion = EMOTIONS[pred.item()]
                confidence = conf.item()*100

                # Green Box
                cv2.rectangle(
                    img,
                    (x,y),
                    (x+bw,y+bh),
                    (0,255,0),
                    2
                )

                text = f"{emotion} {confidence:.1f}%"

                cv2.putText(
                    img,
                    text,
                    (x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0,255,0),
                    2
                )

        return av.VideoFrame.from_ndarray(img, format="bgr24")

import streamlit as st


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

