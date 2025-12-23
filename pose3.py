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
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    angle = np.degrees(angle)
    
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

# Improved UI drawing function
def draw_ui(image, feedback, feedback_color, accuracy, counter_bicep, counter_squat, counter_lateral_raise, stage_bicep, stage_squat, stage_lateral_raise, bicep_angle =0, squat_angle =0, lateral_raise_angle=0):
    # Get image dimensions
    height, width, _ = image.shape
    
    # Create a semi-transparent overlay for UI elements
    overlay = image.copy()
    
    # Top bar overlay
    cv2.rectangle(overlay, (0, 0), (width, 80), (20, 20, 20), -1)
    
    # Side panel overlay (right side)
    cv2.rectangle(overlay, (width-300, 80), (width, height), (20, 20, 20), -1)
    
    # Bottom bar overlay
    cv2.rectangle(overlay, (0, height-100), (width-300, height), (20, 20, 20), -1)
    
    # Apply transparency
    alpha = 0.7  # Higher alpha = more transparent
    cv2.addWeighted(overlay, 1-alpha, image, alpha, 0, image)
    
    # Draw divider lines
    cv2.line(image, (width-300, 80), (width-300, height), (100, 100, 100), 2)
    cv2.line(image, (0, 80), (width-300, 80), (100, 100, 100), 2)
    cv2.line(image, (0, height-100), (width-300, height-100), (100, 100, 100), 2)
    
    # Top header - Title and current exercise
    cv2.putText(image, "Exercise Form Analyzer", (20, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(image, f"Current: {exercise}", (20, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Accuracy on top-right
    cv2.putText(image, f"Confidence: {accuracy:.2f}%", (width-290, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Feedback indicator - Color circle based on form correctness
    cv2.circle(image, (width-40, 40), 15, feedback_color, -1)
    
    # Side panel - Exercise stats
    panel_x = width - 280
    
    # Bicep curl stats
    cv2.putText(image, "BICEP CURL", (panel_x, 110), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(image, f"Reps: {counter_bicep}", (panel_x, 140), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(image, f"Stage: {stage_bicep if stage_bicep else 'N/A'}", (panel_x, 170), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Squat stats
    cv2.putText(image, "SQUAT", (panel_x, 220), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(image, f"Reps: {counter_squat}", (panel_x, 250), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(image, f"Stage: {stage_squat if stage_squat else 'N/A'}", (panel_x, 280), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Lateral raise stats
    cv2.putText(image, "LATERAL RAISE", (panel_x, 330), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(image, f"Reps: {counter_lateral_raise}", (panel_x, 360), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(image, f"Stage: {stage_lateral_raise if stage_lateral_raise else 'N/A'}", (panel_x, 390), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Feedback text - bottom bar
    cv2.putText(image, f"Form Feedback: {feedback}", (20, height-60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Add indicator of correct form in bottom right
    indicator_color = (0, 255, 0) if feedback == "Correct Form" else (0, 0, 255)
    cv2.circle(image, (width-320, height-60), 10, indicator_color, -1)
    
    # Angles display in bottom bar - smaller and positioned to not overlap with exercise view
    cv2.putText(image, f"Bicep: {int(bicep_angle)}°", (20, height-30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(image, f"Squat: {int(squat_angle)}°", (180, height-30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(image, f"Lateral: {int(lateral_raise_angle)}°", (340, height-30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Key controls hint
    cv2.putText(image, "Press 'q' to quit", (width-280, height-30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
    return image

# Define absolute path for project folder
PROJECT_ROOT = r"C:\Users\manan\Desktop\Projects\RepBot"
os.makedirs(PROJECT_ROOT, exist_ok=True)
os.chdir(PROJECT_ROOT)

# Confirm the new working directory
print("Updated Working Directory:", os.getcwd())

# Open webcam - Set increased resolution for better visibility
cap = cv2.VideoCapture(0)
# Set larger resolution - adjust as needed for your webcam capabilities
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Counter variables for exercises
counter_bicep = 0
counter_squat = 0
counter_lateral_raise = 0
stage_bicep = None
stage_squat = None
stage_lateral_raise = None
exercise = "None"

# Create CSV file for logging exercise data
csv_path = os.path.join(PROJECT_ROOT, 'exercise_data.csv')
with open(csv_path, mode='w', newline='') as file:
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
        accuracy = 0
        try:
            landmarks = results.pose_landmarks.landmark
            confidence_scores = [landmark.visibility for landmark in landmarks]
            accuracy = np.mean(confidence_scores) * 100  # Convert to percentage
            
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
                correct = True # Default

            # Feedback text and color
            if correct:
                feedback = "Correct Form"
                feedback_color = (0, 255, 0)  # Green
            else:
                feedback = "Incorrect Form"
                feedback_color = (0, 0, 255)  # Red

            # Determine exercise form for logging
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

        except Exception as e:
            print(f"Error: {e}")
            # Set default values if landmarks detection fails
            angle_bicep = 0
            angle_squat = 0
            angle_lateral_raise = 0
            feedback = "No pose detected"
            feedback_color = (0, 165, 255)  # Orange
        
        # Draw skeletal landmarks with improved visibility
        if results.pose_landmarks:
            # Draw skeleton with more visible colors
            drawing_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
            connection_spec = mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2)
            mp_drawing.draw_landmarks(
                image, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS,
                drawing_spec,
                connection_spec
            )
        
        # Render the UI with all components
        image = draw_ui(image, feedback, feedback_color, accuracy, counter_bicep, counter_squat, counter_lateral_raise, 
              stage_bicep, stage_squat, stage_lateral_raise, angle_bicep, angle_squat, angle_lateral_raise)        
        # Create window with adjustable size
        cv2.namedWindow('Exercise Feedback System', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Exercise Feedback System', 1280, 720)  # Set window size
        cv2.imshow('Exercise Feedback System', image)

        # Exit condition
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()

# Preprocess the data (after data collection)
print("\nProcessing collected exercise data...")

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

# Save the trained model using joblib
model_path = os.path.join(PROJECT_ROOT, 'exercise_form_model.pkl')
scaler_path = os.path.join(PROJECT_ROOT, 'scaler.pkl')

# Dump the model and scaler in the AI Project folder
joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)
print(f"\nModel saved to: {model_path}")
print(f"Scaler saved to: {scaler_path}")