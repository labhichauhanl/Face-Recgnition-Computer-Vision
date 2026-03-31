import cv2
from mood_recognition import MoodRecognizer
import csv
import datetime
import os

def main():
    recognizer = MoodRecognizer()

    # Open camera feed (0 for default camera)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Press 'q' to quit, 's' to save mood log.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Analyze mood for all faces in the frame
        mood_results = recognizer.analyze_mood(frame)

        # For each detected face and mood
        for mood_result in mood_results:
            # Get face region
            region = mood_result.get('region', {})
            x = region.get('x', 0)
            y = region.get('y', 0)
            w = region.get('w', 0)
            h = region.get('h', 0)

            # Draw rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            dominant_mood = mood_result.get('dominant_emotion', 'unknown')
            confidence = recognizer.get_confidence(mood_result)

            # Display mood on frame
            label = f"{dominant_mood}: {confidence:.2f}"
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Display the frame
        cv2.imshow('Face/Mood Recognition', frame)

        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save mood log for all detected faces
            if mood_results:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'logs.csv')
                os.makedirs(os.path.dirname(log_path), exist_ok=True)
                with open(log_path, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    for mood_result in mood_results:
                        dominant_mood = mood_result.get('dominant_emotion', 'unknown')
                        confidence = recognizer.get_confidence(mood_result)
                        writer.writerow([timestamp, dominant_mood, confidence])
                print(f"Logged moods for {len(mood_results)} face(s)")

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
