import av
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer

from rps_logic import random_play

st.title("Game Rock-Paper-Scissors")

st.write(random_play.play())

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

mdc = st.sidebar.slider("Detection confidence", 0.0, 1.0, value=0.5, step=0.1)
mtc = st.sidebar.slider("Tracking confidence", 0.0, 1.0, value=0.5, step=0.1)


def track_hands_in_videos(image, min_detection_confidence=mdc, min_tracking_confidence=mtc):
    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
        static_image_mode=False,
    ) as hands:
        image = cv2.flip(image, 1)
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        blank_annotated_image = np.zeros(image.shape, np.uint8)
        blank_annotated_image[:, :, :] = (255, 255, 255)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(blank_annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    return blank_annotated_image


class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        annotated_image = track_hands_in_videos(img)

        return av.VideoFrame.from_ndarray(annotated_image, format="bgr24")


webrtc_streamer(
    key="RockPaperScissor",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
)
