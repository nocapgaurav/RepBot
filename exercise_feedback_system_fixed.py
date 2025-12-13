
import cv2
import mediapipe as mp
import numpy as np
import math
import csv
import time
from sklearn.ensemble import RandomForestClassifier

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def calculate_distance(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.linalg.norm(a - b)

cap = cv2.VideoCapture(0)

counter = 0
stage = None
form = 0
feedback = "Fix Form"
feedback_color = (0, 0, 255)
angles = []
distances = []
data = []

file_path = "exercise_data.csv"
with open(file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["left_elbow_angle", "left_shoulder_angle", "right_elbow_angle", "right_shoulder_angle",
                     "left_wrist_shoulder_dist", "right_wrist_shoulder_dist", "class"])

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    start_time = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Ignoring empty camera frame.")
            continue

        feedback = "No Pose Detected"
        feedback_color = (128, 128, 128)  # Neutral gray

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark

            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOICE.value].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            left_shoulder_angle = calculate_angle(left_elbow, left_shoulder,
                                                  [left_shoulder[0], left_shoulder[1] - 0.1])
            right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
            right_shoulder_angle = calculate_angle(right_elbow, right_shoulder,
                                                   [right_shoulder[0], right_shoulder[1] - 0.1])

            left_wrist_shoulder_dist = calculate_distance(left_wrist, left_shoulder)
            right_wrist_shoulder_dist = calculate_distance(right_wrist, right_shoulder)

            angles.append([left_elbow_angle, left_shoulder_angle,
                           right_elbow_angle, right_shoulder_angle])
            distances.append([left_wrist_shoulder_dist, right_wrist_shoulder_dist])

            if left_elbow_angle > 160 and right_elbow_angle > 160:
                stage = "down"
                feedback = "Go Up"
                feedback_color = (0, 255, 255)
            if left_elbow_angle < 30 and right_elbow_angle < 30 and stage == "down":
                stage = "up"
                counter += 1
                form = 1
                feedback = "Good Rep"
                feedback_color = (0, 255, 0)
            elif stage == "up" and (left_elbow_angle > 30 or right_elbow_angle > 30):
                feedback = "Fix Form"
                feedback_color = (0, 0, 255)

            with open(file_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([left_elbow_angle, left_shoulder_angle, right_elbow_angle,
                                 right_shoulder_angle, left_wrist_shoulder_dist,
                                 right_wrist_shoulder_dist, feedback])

        except Exception as e:
            print("Pose landmarks not detected:", e)

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.rectangle(image, (0, 0), (300, 100), (245, 117, 16), -1)
        cv2.putText(image, 'REPS', (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(image, str(counter), (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)

        cv2.putText(image, feedback, (100, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, feedback_color, 2)

        cv2.imshow('Exercise Feedback', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if angles and distances:
        angles = np.array(angles)
        distances = np.array(distances)
        features = np.hstack((angles, distances))
        labels = np.array([1 if f == "Good Rep" else 0 for f in feedback])
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(features, labels)
        print("Model trained on collected data.")
