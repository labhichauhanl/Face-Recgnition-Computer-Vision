import cv2
import numpy as np

class FaceDetector:
    def __init__(self):
        # Load Haar cascade for face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def detect_faces(self, frame):
        """
        Detect faces in a given frame.
        Returns: list of (x, y, w, h) tuples for face bounding boxes.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Equalize histogram for better detection in varying lighting
        gray = cv2.equalizeHist(gray)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return faces

    def preprocess_face(self, frame, face_rect):
        """
        Preprocess a detected face for further analysis.
        face_rect: (x, y, w, h)
        Returns: preprocessed face image (grayscale, resized).
        """
        x, y, w, h = face_rect
        face = frame[y:y+h, x:x+w]
        # Convert to grayscale
        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        # Resize to standard size (e.g., 48x48 for emotion models, but adjustable)
        face_resized = cv2.resize(face_gray, (48, 48), interpolation=cv2.INTER_AREA)
        return face_resized
