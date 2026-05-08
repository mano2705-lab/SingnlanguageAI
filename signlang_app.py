import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import joblib
import pyttsx3
import tempfile
import time

# Load model and encoder
model = joblib.load("sign_classifier.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Initialize TTS engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1.0)

# MediaPipe Hands setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Streamlit UI
st.title("🧠 SignLang AI – Real-Time Sign Language to Text + Speech")
st.markdown("Show a trained hand sign to your webcam and hear it spoken out loud.")
start = st.button("Start Recognition")
stop_signal = False

# Video display
frame_window = st.image([])

# Control speech delay
last_spoken = ""
last_spoken_time = 0
speak_delay = 1.5  # seconds

if start:
    cap = cv2.VideoCapture(0)
    st.info("Press **Stop** or close tab to exit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        sign_text = ""

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get landmarks
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])

                if len(landmarks) == 63:
                    prediction = model.predict([landmarks])[0]
                    sign_text = label_encoder.inverse_transform([prediction])[0]

                    # Speak with delay
                    current_time = time.time()
                    if sign_text != last_spoken or (current_time - last_spoken_time) > speak_delay:
                        engine.say(sign_text)
                        engine.runAndWait()
                        last_spoken = sign_text
                        last_spoken_time = current_time

        # Display prediction
        if sign_text:
            cv2.putText(frame, f'Sign: {sign_text}', (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
    cv2.destroyAllWindows()
