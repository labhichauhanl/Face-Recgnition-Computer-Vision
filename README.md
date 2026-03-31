# Face-Recgnition-Computer-Vision
A computer vision capstone project for real-time face detection and mood analysis to monitor student engagement and mental health in virtual classrooms.
# Problem Statement
Instructors in online classrooms struggle to assess real-time student engagement and emotional states, such as boredom or confusion, leading to ineffective teaching and missed opportunities for intervention. This system automatically detects faces for attendance tracking and analyzes facial expressions to classify moods (e.g., happy, sad, angry, neutral), logging data for review and enabling educators to provide timely support.

# Installation
1. Clone the repository:

git clone https://github.com/yourusername/CV-project.git
cd CV-project

2. Install dependencies:

pip install -r requirements.txt
# Usage
Run the main script to start the camera feed:

python src/main.py
Faces are detected and highlighted in blue rectangles.
Dominant mood and confidence score are displayed above each face.
Press 's' to save the current mood data to data/logs.csv.
Press 'q' to quit.
# Features
Real-time face detection using OpenCV Haar cascades.
Mood recognition via DeepFace (pre-trained emotion model).
CSV logging for mood tracking with timestamps.
Histogram equalization for improved detection in varying lighting.
# Demo
[Insert GIF/screen recording here showing face detection and mood analysis in action.]

# Tech Stack
OpenCV: Face detection and image preprocessing.
DeepFace: Emotion analysis with TensorFlow backend.
Python: Core scripting with NumPy and Pandas for data handling.
# License
MIT License
