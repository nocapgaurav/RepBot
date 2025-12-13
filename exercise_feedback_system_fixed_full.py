
import numpy as np
import cv2
import mediapipe as mp
import csv
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import pandas as pd

# UI
def draw_ui(image, feedback, feedback_color, accuracy, counter_bicep, counter_squat, counter_lateral_raise, stage_bicep, stage_squat, stage_lateral_raise):
    height, width, _ = image.shape
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 0), -1)
    alpha = 0.4
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    cv2.rectangle(image, (50, 380), (width - 50, 430), feedback_color, -1)
    cv2.putText(image, f"Form Feedback: {feedback}", (60, 415), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
    cv2.putText(image, f"Confidence: {accuracy:.2f}%", (width - 250, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    start_x = 50
    cv2.rectangle(image, (start_x, 450), (width - 50, 500), (50, 50, 50), -1)
    cv2.putText(image, "Exercise", (start_x + 20, 485), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(image, "Reps", (start_x + 200, 485), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(image, "Stage", (start_x + 350, 485), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(image, "Accuracy", (start_x + 500, 485), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    exercises = [("Bicep Curl", counter_bicep, stage_bicep, accuracy),
                 ("Squat", counter_squat, stage_squat, accuracy),
                 ("Lateral Raise", counter_lateral_raise, stage_lateral_raise, accuracy)]

    start_y = 510
    for ex_name, reps, stage, acc in exercises:
        cv2.rectangle(image, (start_x, start_y), (width - 50, start_y + 40), (30, 30, 30), -1)
        cv2.putText(image, ex_name, (start_x + 20, start_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, str(reps), (start_x + 220, start_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, stage if stage else "N/A", (start_x + 370, start_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, f"{acc:.2f}%", (start_x + 540, start_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        start_y += 50

    return image

PROJECT_ROOT = r"C:\Users\ASUS IND\OneDrive\Desktop\AI PROJECT"
os.makedirs(PROJECT_ROOT, exist_ok=True)
os.chdir(PROJECT_ROOT)
print("Updated Working Directory:", os.getcwd())

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def exercise_form_is_correct(bicep_angle, squat_angle, lateral_raise_angle):
    if not (30 < bicep_angle < 160): return False
    if not (90 < squat_angle < 160): return False
    if lateral_raise_angle >= 90: return False
    return True

cap = cv2.VideoCapture(0)
counter_bicep = counter_squat = counter_lateral_raise = 0
stage_bicep = stage_squat = stage_lateral_raise = None
exercise = "None"
file_path = os.path.join(PROJECT_ROOT, 'exercise_data.csv')
with open(file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Bicep Angle', 'Squat Angle', 'Lateral Raise Angle', 'Label'])

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        accuracy = 0
        feedback, feedback_color = "No Feedback", (255, 255, 255)

        if results.pose_landmarks:
            try:
                landmarks = results.pose_landmarks.landmark
                confidence_scores = [lm.visibility for lm in landmarks]
                accuracy = np.mean(confidence_scores) * 100
                print(f"Pose Estimation Accuracy: {accuracy:.2f}%")

                def get_coords(part):
                    lm = landmarks[mp_pose.PoseLandmark[part].value]
                    return [lm.x, lm.y]

                shoulder, elbow, wrist = get_coords("LEFT_SHOULDER"), get_coords("LEFT_ELBOW"), get_coords("LEFT_WRIST")
                hip, knee, ankle = get_coords("LEFT_HIP"), get_coords("LEFT_KNEE"), get_coords("LEFT_ANKLE")
                shoulder_left, shoulder_right, elbow_left = get_coords("LEFT_SHOULDER"), get_coords("RIGHT_SHOULDER"), get_coords("LEFT_ELBOW")

                angle_bicep = calculate_angle(shoulder, elbow, wrist)
                angle_squat = calculate_angle(hip, knee, ankle)
                angle_lateral_raise = calculate_angle(shoulder_left, elbow_left, shoulder_right)

                is_correct_form = exercise_form_is_correct(angle_bicep, angle_squat, angle_lateral_raise)
                feedback = "Correct Form" if is_correct_form else "Incorrect Form"
                feedback_color = (0, 255, 0) if is_correct_form else (0, 0, 255)

                label = "correct" if is_correct_form else "incorrect"
                with open('exercise_data.csv', mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([angle_bicep, angle_squat, angle_lateral_raise, label])

                if angle_bicep > 160: stage_bicep = "down"
                if angle_bicep < 30 and stage_bicep == "down":
                    stage_bicep = "up"; counter_bicep += 1; exercise = "Bicep Curl"

                if angle_squat > 160: stage_squat = "up"
                if angle_squat < 90 and stage_squat == "up":
                    stage_squat = "down"; counter_squat += 1; exercise = "Squat"

                if angle_lateral_raise > 160: stage_lateral_raise = "down"
                if angle_lateral_raise < 30 and stage_lateral_raise == "down":
                    stage_lateral_raise = "up"; counter_lateral_raise += 1; exercise = "Lateral Raise"

            except Exception as e:
                print(f"Error: {e}")

        cv2.rectangle(image, (0, 0), (640, 50), (0, 0, 0), -1)
        cv2.putText(image, f"Exercise: {exercise}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.rectangle(image, (0, 50), (213, 150), (50, 50, 50), -1)
        cv2.putText(image, 'Bicep Curl', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, f"Reps: {counter_bicep}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, f"Stage: {stage_bicep}", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.rectangle(image, (213, 50), (426, 150), (50, 50, 50), -1)
        cv2.putText(image, 'Squat', (233, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, f"Reps: {counter_squat}", (233, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, f"Stage: {stage_squat}", (233, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.rectangle(image, (426, 50), (640, 150), (50, 50, 50), -1)
        cv2.putText(image, 'Lateral Raise', (446, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, f"Reps: {counter_lateral_raise}", (446, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, f"Stage: {stage_lateral_raise}", (446, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
        image = draw_ui(image, feedback, feedback_color, accuracy, counter_bicep, counter_squat, counter_lateral_raise, stage_bicep, stage_squat, stage_lateral_raise)
        cv2.imshow('Exercise Feedback System', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

data = pd.read_csv('exercise_data.csv')
data.dropna(inplace=True)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[['Bicep Angle', 'Squat Angle', 'Lateral Raise Angle']])
encoder = LabelEncoder()
encoded_labels = encoder.fit_transform(data['Label'])
X_train, X_test, y_train, y_test = train_test_split(scaled_data, encoded_labels, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on Test Data: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=encoder.classes_))
print("Current Working Directory:", os.getcwd())

model_path = os.path.join(PROJECT_ROOT, 'exercise_form_model.pkl')
scaler_path = os.path.join(PROJECT_ROOT, 'scaler.pkl')
joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)
