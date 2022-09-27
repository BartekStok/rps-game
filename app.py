import av
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

from const import TransformationTypes

st.title("Game Rock-Paper-Scissors")

# st.write(random_play.play())

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

MDC = st.sidebar.slider("Detection confidence", 0.0, 1.0, value=0.5, step=0.1)
MTC = st.sidebar.slider("Tracking confidence", 0.0, 1.0, value=0.5, step=0.1)


def track_hands_in_videos(image, min_detection_confidence, min_tracking_confidence):
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


def track_fingers_in_videos(image, min_detection_confidence, min_tracking_confidence):
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
        # if results.multi_hand_landmarks:
        #     for hand_landmarks in results.multi_hand_landmarks:
        #         mp_drawing.draw_landmarks(blank_annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        point = np.zeros((20, 40, 3), np.uint8)
        point[10, 5] = [0, 0, 255]
        # cv2.imwrite(blank_annotated_image, point)
        blank_annotated_image[100, 100] = [0, 0, 255]
        print(blank_annotated_image.shape)

    return blank_annotated_image


class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.type = TransformationTypes.normal.value

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        annotated_image = None

        if self.type == TransformationTypes.hands.value:
            annotated_image = track_hands_in_videos(img, MDC, MTC)

        if self.type == TransformationTypes.fingers.value:
            annotated_image = track_fingers_in_videos(img, MDC, MTC)

        return av.VideoFrame.from_ndarray(annotated_image, format="bgr24")


if __name__ == "__main__":
    # COMMON_RTC_CONFIG = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    COMMON_RTC_CONFIG = None
    TRANSFORMATION_TYPES = TransformationTypes.set()

    ctx = webrtc_streamer(
        key="loopback",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=COMMON_RTC_CONFIG,
        media_stream_constraints={
            "video": True,
            "audio": False,
        },
    )

    filter1_ctx = webrtc_streamer(
        key="filter1",
        mode=WebRtcMode.RECVONLY,
        video_processor_factory=VideoProcessor,
        source_video_track=ctx.output_video_track,
        desired_playing_state=ctx.state.playing,
        rtc_configuration=COMMON_RTC_CONFIG,
        media_stream_constraints={
            "video": True,
            "audio": False,
        },
    )

    if filter1_ctx.video_processor:
        filter1_ctx.video_processor.type = st.radio(
            "Select transform type",
            TRANSFORMATION_TYPES,
            key="filter1-type",
        )

    filter2_ctx = webrtc_streamer(
        key="filter2",
        mode=WebRtcMode.RECVONLY,
        video_processor_factory=VideoProcessor,
        source_video_track=ctx.output_video_track,
        desired_playing_state=ctx.state.playing,
        rtc_configuration=COMMON_RTC_CONFIG,
        media_stream_constraints={
            "video": True,
            "audio": False,
        },
    )
    if filter2_ctx.video_processor:
        filter2_ctx.video_processor.type = st.radio(
            "Select transform type",
            TRANSFORMATION_TYPES,
            key="filter2-type",
        )


line = st.empty()
