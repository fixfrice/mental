# ai_models/yolo_detector.py

import os
from deepface import DeepFace

def predict(image_path):
    if not os.path.exists(image_path):
        return "Image not found."

    try:
        # Analyze the face and detect emotion
        result = DeepFace.analyze(image_path, actions=["emotion"], enforce_detection=False)

        # Extract the dominant emotion
        emotion = result[0]['dominant_emotion']

        # Map emotions to mental states
        emotion_to_state = {
            "happy": "Relaxed",
            "neutral": "Calm",
            "sad": "Tired",
            "angry": "Irritated",
            "fear": "Anxious",
            "disgust": "Stressed",
            "surprise": "Excited / Tense"
        }

        # Get mental state by emotion
        state = emotion_to_state.get(emotion.lower(), "Undefined")

        return f"Mental state detected: {state} (emotion: {emotion})"

    except Exception as e:
        return f"Error analyzing image: {str(e)}"
