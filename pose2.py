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
import joblib
import pandas as pd


#UI
# def draw_ui(image, feedback, feedback_color, accuracy, counter_bicep, counter_squat, counter_lateral_raise, stage_bicep, stage_squat, stage_lateral_raise):
#     height, width, _ = image.shape

#     # Dark transparent overlay for a modern look
#     overlay = image.copy()
#     cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 0), -1)  
#     alpha = 0.4  # Transparency factor
#     cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

#     # Feedback box
#     cv2.rectangle(image, (50, 380), (width - 50, 430), feedback_color, -1)
#     cv2.putText(image, f"Form Feedback: {feedback}", (60, 415), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

#     # Confidence Score
#     cv2.putText(image, f"Confidence: {accuracy:.2f}%", (width - 250, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

#     # Exercise table
#     start_x = 50
#     cv2.rectangle(image, (start_x, 450), (width - 50, 500), (50, 50, 50), -1)  # Header
#     cv2.putText(image, "Exercise", (start_x + 20, 485), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
#     cv2.putText(image, "Reps", (start_x + 200, 485), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
#     cv2.putText(image, "Stage", (start_x + 350, 485), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
#     cv2.putText(image, "Accuracy", (start_x + 500, 485), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

#     exercises = [("Bicep Curl", counter_bicep, stage_bicep, accuracy),
#                  ("Squat", counter_squat, stage_squat, accuracy),
#                  ("Lateral Raise", counter_lateral_raise, stage_lateral_raise, accuracy)]
    
#     start_y = 510
#     for ex_name, reps, stage, acc in exercises:
#         cv2.rectangle(image, (start_x, start_y), (width - 50, start_y + 40), (30, 30, 30), -1)
#         cv2.putText(image, ex_name, (start_x + 20, start_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
#         cv2.putText(image, str(reps), (start_x + 220, start_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
#         cv2.putText(image, stage if stage else "N/A", (start_x + 370, start_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
#         cv2.putText(image, f"{acc:.2f}%", (start_x + 540, start_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
#         start_y += 50  # Move to next row

#     return image


def draw_ui(image, feedback, feedback_color, accuracy, counter_bicep, counter_squat, counter_lateral_raise, stage_bicep, stage_squat, stage_lateral_raise):
    height, width, _ = image.shape
    
    # Calculate dynamic margins and spaces based on image size
    margin_x = int(width * 0.05)  # 5% margin on both sides
    margin_y = int(height * 0.05)
    
    # Dark transparent overlay for a modern look
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 0), -1)  
    alpha = 0.4  # Transparency factor
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    # Feedback box - positioned with proper margins and padding
    feedback_y_start = height - 200
    feedback_box_height = 60
    cv2.rectangle(image, (margin_x, feedback_y_start), (width - margin_x, feedback_y_start + feedback_box_height), feedback_color, -1)
    cv2.putText(image, f"Form Feedback: {feedback}", (margin_x + 10, feedback_y_start + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Confidence Score - positioned at the top right
    cv2.putText(image, f"Confidence: {accuracy:.2f}%", (width - 250, margin_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Exercise table - positioned below feedback box with proper spacing
    table_y_start = feedback_y_start + feedback_box_height + 10
    header_height = 50
    row_height = 40
    col_widths = [180, 100, 150, 150]  # Column widths
    
    # Calculate total table width
    table_width = sum(col_widths)
    table_x_start = (width - table_width) // 2  # Center the table horizontally

    # Draw table header
    header_color = (50, 50, 50)
    current_x = table_x_start
    
    # Header background
    cv2.rectangle(image, (table_x_start, table_y_start), 
                 (table_x_start + table_width, table_y_start + header_height), 
                 header_color, -1)
    
    # Header text
    headers = ["Exercise", "Reps", "Stage", "Accuracy"]
    for i, header in enumerate(headers):
        text_x = current_x + (col_widths[i] // 2) - (len(header) * 5)  # Center text in column
        cv2.putText(image, header, (text_x, table_y_start + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        current_x += col_widths[i]

    # Draw table rows
    exercises = [
        ("Bicep Curl", counter_bicep, stage_bicep, accuracy),
        ("Squat", counter_squat, stage_squat, accuracy),
        ("Lateral Raise", counter_lateral_raise, stage_lateral_raise, accuracy)
    ]
    
    row_y = table_y_start + header_height
    row_color = (30, 30, 30)
    
    for ex_name, reps, stage, acc in exercises:
        # Row background
        cv2.rectangle(image, (table_x_start, row_y), 
                     (table_x_start + table_width, row_y + row_height), 
                     row_color, -1)
        
        # Row data
        current_x = table_x_start
        
        # Exercise name
        text_x = current_x + 10  # Left-aligned
        cv2.putText(image, ex_name, (text_x, row_y + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        current_x += col_widths[0]
        
        # Reps
        text_x = current_x + (col_widths[1] // 2) - 10  # Center-aligned
        cv2.putText(image, str(reps), (text_x, row_y + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        current_x += col_widths[1]
        
        # Stage
        text_x = current_x + 10  # Left-aligned
        stage_text = stage if stage else "N/A"
        cv2.putText(image, stage_text, (text_x, row_y + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        current_x += col_widths[2]
        
        # Accuracy
        text_x = current_x + (col_widths[3] // 2) - 25  # Center-aligned
        cv2.putText(image, f"{acc:.2f}%", (text_x, row_y + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        row_y += row_height  # Move to next row
    
    # Draw title at the top
    # cv2.putText(image, "Exercise Form Analyzer", (width // 2 - 150, margin_y), 
    #            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    
    return image


# Define project root as script directory (cross-platform)
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
os.makedirs(PROJECT_ROOT, exist_ok=True)
print("Using PROJECT_ROOT:", PROJECT_ROOT)


# Initialize Mediapipe Pose model
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # Point A
    b = np.array(b)  # Point B
    c = np.array(c)  # Point C
    
    ba = a - b
    bc = c - b
    # Protect against zero-length vectors and floating point errors
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba == 0 or norm_bc == 0:
        return 0.0

    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    # Clamp cosine to valid range to avoid NaNs from floating point errors
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine_angle))
    return angle

# Function to check if the exercise form is correct
def exercise_form_is_correct(bicep_angle, squat_angle, lateral_raise_angle):
    # Define rules for correct form
    # Bicep curl: 30 to 160 degrees
    if not (30 < bicep_angle < 160):
        return False
    # Squat: 90 to 160 degrees
    if not (90 < squat_angle < 160):
        return False
    # Lateral raise: less than 90 degrees
    if lateral_raise_angle >= 90:
        return False
    return True

# Open webcam
cap = cv2.VideoCapture(0)

# Curl counter variables
counter_bicep = 0
counter_squat = 0
counter_lateral_raise = 0
stage_bicep = None
stage_squat = None
stage_lateral_raise = None
exercise = "None"

csv_path = os.path.join(PROJECT_ROOT, 'exercise_data.csv')

# Create CSV file for logging exercise data (only once at the start)
file_path = csv_path
with open(file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Bicep Angle', 'Squat Angle', 'Lateral Raise Angle', 'Label'])  # header

# Setup Mediapipe Pose model
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Pose detection
        results = pose.process(image)
        
        # Convert back to BGR for OpenCV rendering
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks and calculate accuracy
        # Default values to ensure UI has something to render
        accuracy = 0.0
        feedback = "No Pose Detected"
        feedback_color = (0, 0, 255)
        # default coords/angles (safe fallbacks)
        elbow = [0, 0]
        knee = [0, 0]
        shoulder_left = [0, 0]
        angle_bicep = 0.0
        angle_squat = 0.0
        angle_lateral_raise = 0.0

        try:
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                confidence_scores = [landmark.visibility for landmark in landmarks]
                if len(confidence_scores) > 0:
                    accuracy = np.mean(confidence_scores) * 100  # Convert to percentage
                print(f"Pose Estimation Accuracy: {accuracy:.2f}%")

                # Get coordinates for bicep curl
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                # Get coordinates for squat
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                # Get coordinates for lateral raise using LEFT arm
                shoulder_left = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                 landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow_left = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist_left = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                # Calculate angle for bicep curl
                angle_bicep = calculate_angle(shoulder, elbow, wrist)

                # Calculate angle for squat
                angle_squat = calculate_angle(hip, knee, ankle)

                # Calculate lateral raise angle: shoulder → elbow → wrist
                angle_lateral_raise = calculate_angle(shoulder_left, elbow_left, wrist_left)

                # Form feedback based on current exercise
                if exercise == "Bicep Curl":
                    correct = 30 < angle_bicep < 160
                elif exercise == "Squat":
                    correct = 90 < angle_squat < 160
                elif exercise == "Lateral Raise":
                    correct = angle_lateral_raise < 90
                else:
                    correct = True  # Default

                # Feedback text and color
                if correct:
                    feedback = "Correct Form"
                    feedback_color = (0, 255, 0)
                else:
                    feedback = "Incorrect Form"
                    feedback_color = (0, 0, 255)

                # Visualize angles only when we have valid landmarks
                elbow_coord = tuple(np.multiply(elbow, [640, 480]).astype(int))
                cv2.putText(image, f"Bicep Angle: {int(angle_bicep)}",
                            elbow_coord,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                knee_coord = tuple(np.multiply(knee, [640, 480]).astype(int))
                cv2.putText(image, f"Squat Angle: {int(angle_squat)}",
                            knee_coord,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                shoulder_coord = tuple(np.multiply(shoulder_left, [640, 480]).astype(int))
                cv2.putText(image, f"Lateral Raise Angle: {int(angle_lateral_raise)}",
                            shoulder_coord,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                # Determine exercise form label
                label = "correct" if exercise_form_is_correct(angle_bicep, angle_squat, angle_lateral_raise) else "incorrect"

                # Log the angles and form status to CSV
                with open(csv_path, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([angle_bicep, angle_squat, angle_lateral_raise, label])

                # Bicep curl counter logic
                if angle_bicep > 160:
                    stage_bicep = "down"
                if angle_bicep < 30 and stage_bicep == "down":
                    stage_bicep = "up"
                    counter_bicep += 1
                    exercise = "Bicep Curl"
                    print(f"Bicep Reps: {counter_bicep}")

                # Squat counter logic
                if angle_squat > 160:
                    stage_squat = "up"
                if angle_squat < 90 and stage_squat == "up":
                    stage_squat = "down"
                    counter_squat += 1
                    exercise = "Squat"
                    print(f"Squat Reps: {counter_squat}")

                # Lateral raise counter logic
                if angle_lateral_raise > 80:
                    stage_lateral_raise = "up"
                if angle_lateral_raise < 40 and stage_lateral_raise == "up":
                    stage_lateral_raise = "down"
                    counter_lateral_raise += 1
                    exercise = "Lateral Raise"
                    print(f"Lateral Raise Reps: {counter_lateral_raise}")

            # else: no landmarks detected; keep defaults
        except Exception as e:
            print(f"Error while processing landmarks: {e}")

        # Display accuracy on screen
        cv2.putText(image, f"Accuracy: {accuracy:.2f}%", (500, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Render exercise UI with improved layout
        # Background rectangle for exercise name
        cv2.rectangle(image, (0, 0), (640, 50), (0, 0, 0), -1)
        cv2.putText(image, f"Exercise: {exercise}", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Bicep Curl Reps and Stage
        cv2.rectangle(image, (0, 50), (213, 150), (50, 50, 50), -1)
        cv2.putText(image, 'Bicep Curl', (20, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, f"Reps: {counter_bicep}", (20, 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, f"Stage: {stage_bicep}", (20, 140), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Squat Reps and Stage
        cv2.rectangle(image, (213, 50), (426, 150), (50, 50, 50), -1)
        cv2.putText(image, 'Squat', (233, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, f"Reps: {counter_squat}", (233, 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, f"Stage: {stage_squat}", (233, 140), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Lateral Raise Reps and Stage
        cv2.rectangle(image, (426, 50), (640, 150), (50, 50, 50), -1)
        cv2.putText(image, 'Lateral Raise', (446, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, f"Reps: {counter_lateral_raise}", (446, 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, f"Stage: {stage_lateral_raise}", (446, 140), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # Render pose landmarks if available
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
        
        # Show the output
        image = draw_ui(image, feedback, feedback_color, accuracy, counter_bicep, counter_squat, counter_lateral_raise, stage_bicep, stage_squat, stage_lateral_raise)
        cv2.imshow('Exercise Feedback System', image)


        # Exit condition
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()

# Preprocess the data (after data collection)
# Load the CSV data
data = pd.read_csv(csv_path)

# Remove any rows with missing values (if any)
data.dropna(inplace=True)

# Scale the features (Bicep Angle, Squat Angle, Lateral Raise Angle)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[['Bicep Angle', 'Squat Angle', 'Lateral Raise Angle']])

# Encode the labels (correct, incorrect)
encoder = LabelEncoder()
encoded_labels = encoder.fit_transform(data['Label'])

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(scaled_data, encoded_labels, test_size=0.2, random_state=42)

# Build the machine learning model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model with the training data
model.fit(X_train, y_train)

# Predict on the test data
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on Test Data: {accuracy * 100:.2f}%")

# Print a classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=encoder.classes_))

print("Current Working Directory:", os.getcwd())

# Save the trained model using joblib

model_path = os.path.join(PROJECT_ROOT, 'exercise_form_model.pkl')
scaler_path = os.path.join(PROJECT_ROOT, 'scaler.pkl')
# Dump the model and scaler in the project folder
joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)