# Face/Mood Recognition System
A real-time computer vision application that detects faces and analyzes facial expressions to recognize and log moods. Designed for monitoring student engagement and emotional states in educational settings.

# Overview
This project combines face detection and emotion recognition to provide real-time mood analysis from camera feeds. It automatically detects faces, analyzes their emotional expressions (e.g., happy, sad, angry, neutral, surprised), and logs the data for later analysis. Perfect for educators who want to gauge student engagement and provide timely intervention.

# Features
Real-Time Face Detection: Automatically detects multiple faces in the camera feed
Emotion Analysis: Classifies emotions using pre-trained deep learning models
Confidence Scores: Displays confidence levels for detected emotions
Data Logging: Saves mood data to CSV for analysis and tracking
Lighting Optimization: Applies histogram equalization to handle varying lighting conditions
Live Display: Shows detected faces with bounding boxes and emotion labels
# Project Structure
CV-project/
├── src/
│   ├── main.py                 # Main application entry point
│   ├── face_detection.py       # Face detection module
│   ├── mood_recognition.py     # Emotion analysis module
│   └── __init__.py
├── data/
│   └── logs.csv               # Mood data log file
├── docs/
│   ├── project_report.md      # Detailed project documentation
│   └── README.md
├── requirements.txt            # Python dependencies
├── LICENSE
└── README.md                   # This file
# How It Works
Face Detection: Uses OpenCV with histogram equalization to detect faces in varying lighting conditions
Mood Recognition: Utilizes DeepFace (with VGG-Face and pre-trained emotion models) to analyze facial expressions
Logging: Records detected moods with timestamps to CSV for later analysis
Display: Live feed shows detected faces with emotion labels and confidence scores
# Requirements
Python 3.7+
OpenCV (cv2)
DeepFace
NumPy
TensorFlow/Keras (required by DeepFace)
See requirements.txt for exact versions.

# Installation
Clone or download the project
Install dependencies:
pip install -r requirements.txt
# Usage
Run the main application:

python src/main.py
# Controls
q: Quit the application
s: Save current mood logs to CSV file
# Output
Console: Real-time emotion analysis results
CSV Log: Mood data saved to data/logs.csv with timestamps
# Key Design Decisions
DeepFace Library: Chosen for pre-trained models and high accuracy without extensive training data
Modular Architecture: Separated detection, recognition, and main logic for easy extension
Histogram Equalization: Improves detection accuracy in poor lighting conditions
Confidence Scores: Helps users interpret results contextually
# Technical Challenges & Solutions
Challenge	Solution
Poor lighting detection	Histogram equalization preprocessing
False positive detections	Adjusted detector parameters; added confidence thresholds
Slow processing	Optimized to analyze emotions only on detected faces
Dependency conflicts	Pinned compatible versions in requirements.txt
# Future Improvements
Integrate SQLite database for scalable logging
Add multi-face attendance tracking with unique IDs
Implement real-time alerts for educators based on mood trends
Train custom model for classroom-specific expressions
Add data visualization dashboard for engagement analytics
# License
See LICENSE file for details.

# Notes
This project demonstrates practical application of computer vision in education, balancing technical feasibility with real-world usability for monitoring student engagement and emotional well-being.
