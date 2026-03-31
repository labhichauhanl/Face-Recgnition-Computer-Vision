from deepface import DeepFace
import cv2
import numpy as np

class MoodRecognizer:
    def __init__(self):
        # DeepFace will use pre-trained models for emotion analysis
        pass

    def analyze_mood(self, frame):
        """
        Analyze the mood/emotion from a frame.
        frame: numpy array (BGR)
        Returns: list of dicts with 'dominant_emotion' and 'emotion' scores for each face.
        """
        try:
            # Use DeepFace to analyze the entire frame, it will detect faces automatically
            results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            # results is a list of dicts, one for each face
            return results
        except Exception as e:
            print(f"Error in mood analysis: {e}")
            return []

    def get_confidence(self, emotion_dict):
        """
        Get confidence score for the dominant emotion.
        """
        dominant = emotion_dict.get('dominant_emotion', 'unknown')
        scores = emotion_dict.get('emotion', {})
        return scores.get(dominant, 0.0)
