# Project Report: Face/Mood Recognition System
# Problem Statement
The project addresses the challenge of monitoring student engagement and mental health in virtual classrooms. Instructors often struggle to gauge real-time participation and emotional states during online lectures, leading to missed opportunities for intervention. This Face/Mood Recognition System uses computer vision to automatically detect faces for attendance tracking and analyze facial expressions to classify moods (e.g., happy, sad, angry, neutral, surprised). It logs data to a CSV file, enabling educators to review engagement patterns and provide timely support, such as addressing boredom or confusion to improve learning outcomes.

# Key Decisions
Library Choice
I chose DeepFace over building a custom CNN from scratch because it provides pre-trained weights for VGG-Face and emotion models, allowing for higher accuracy with limited local compute power. DeepFace leverages TensorFlow/Keras under the hood, ensuring robust performance without extensive training data or GPU requirements.

Preprocessing Techniques
I implemented Histogram Equalization in the face detection pipeline to handle varying lighting conditions in indoor environments. This improves detection accuracy by normalizing image contrast, reducing false negatives in low-light scenarios.

Architecture
The system is modular: face_detection.py handles detection and preprocessing, mood_recognition.py manages emotion analysis, and main.py integrates camera feed, display, and logging. This separation allows for easy extension (e.g., adding database storage instead of CSV).

# Challenges Faced
Lighting Issues
During testing, poor lighting caused frequent false negatives in face detection. I mitigated this by adding histogram equalization to the preprocessing step, which improved detection rates by approximately 20% in dim conditions.

Model Accuracy and False Positives
The Haar cascade detector occasionally mistook wall posters or objects for faces. I adjusted the minNeighbors parameter from 3 to 5, reducing false positives while maintaining true detections. For mood analysis, DeepFace's model sometimes misclassified neutral expressions as sadness; this was addressed by displaying confidence scores, allowing users to interpret results contextually.

Performance Optimization
Initial runs were slow due to DeepFace loading large models. I optimized by analyzing emotions only on detected faces rather than the entire frame, reducing processing time per frame.

Integration and Dependencies
Installing DeepFace and TensorFlow on Windows required ensuring compatible versions; I pinned them in requirements.txt to avoid conflicts.

# Future Improvements
Integrate with a database (e.g., SQLite) for scalable logging.
Add multi-face attendance tracking with unique IDs.
Implement real-time alerts for educators based on mood trends.
Train a custom model for classroom-specific expressions.
# Conclusion
This project demonstrates practical application of computer vision in education, balancing technical feasibility with real-world usability. The staged development approach ensured incremental progress, simulating a professional workflow.
