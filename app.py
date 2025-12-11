# # from flask import Flask, render_template, Response
# # from flask_socketio import SocketIO
# # from pose_model import FitnessTrainer, ExerciseType
# # import cv2
# # import numpy as np
# # import time

# # app = Flask(__name__)
# # socketio = SocketIO(app, cors_allowed_origins="*")
# # trainer = FitnessTrainer()

# # def generate_frames():
# #     cap = cv2.VideoCapture(0)
    
# #     if not cap.isOpened():
# #         print("Error: Could not open camera")
# #         while True:
# #             frame = np.zeros((480, 640, 3), dtype=np.uint8)
# #             cv2.putText(frame, "CAMERA ERROR", (50, 240), 
# #                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
# #             ret, buffer = cv2.imencode('.jpg', frame)
# #             yield (b'--frame\r\n'
# #                    b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
# #             time.sleep(0.1)
# #         return

# #     while True:
# #         try:
# #             success, frame = cap.read()
# #             if not success:
# #                 break
            
# #             frame = trainer.process_frame(frame)
            
# #             if trainer.current_exercise != ExerciseType.NONE:
# #                 current_fb = trainer.exercises[trainer.current_exercise]
# #                 stats = {
# #                     "reps": current_fb.counter,
# #                     "feedback": current_fb.feedback,
# #                     "rate": f"{current_fb.rep_rate:.1f}"
# #                 }
# #                 socketio.emit('stats_update', stats)
            
# #             ret, buffer = cv2.imencode('.jpg', frame)
# #             yield (b'--frame\r\n'
# #                    b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
# #         except Exception as e:
# #             print(f"Camera Error: {str(e)}")
# #             break
    
# #     cap.release()

# # @app.route('/')
# # def index():
# #     return render_template('index.html')

# # @app.route('/video_feed')
# # def video_feed():
# #     return Response(
# #         generate_frames(),
# #         mimetype='multipart/x-mixed-replace; boundary=frame',
# #         headers={
# #             'Cache-Control': 'no-cache, no-store, must-revalidate',
# #             'Pragma': 'no-cache',
# #             'Expires': '0'
# #         }
# #     )

# # @socketio.on('set_exercise')
# # def handle_exercise_change(exercise):
# #     exercise_map = {
# #         'bicep': ExerciseType.BICEP_CURL,
# #         'squat': ExerciseType.SQUAT,
# #         'lateral': ExerciseType.LATERAL_RAISE,
# #         'none': ExerciseType.NONE
# #     }
# #     trainer.current_exercise = exercise_map.get(exercise, ExerciseType.NONE)
# #     socketio.emit('exercise_changed', {'exercise': exercise})

# # if __name__ == '__main__':
# #     socketio.run(app, host='127.0.0.1', port=5000, debug=True)
# from flask import Flask, render_template, Response
# from flask_socketio import SocketIO
# from pose_model import FitnessTrainer, ExerciseType
# import cv2
# import numpy as np
# import time

# # Initialize Flask app and SocketIO
# app = Flask(__name__)
# socketio = SocketIO(app, cors_allowed_origins="*")

# # Instantiate the fitness trainer (handles pose processing and stats)
# trainer = FitnessTrainer()

# def generate_frames():
#     """
#     Capture frames from webcam, process with FitnessTrainer,
#     emit stats over SocketIO, and yield JPEG frames as multipart.
#     """
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         # Camera error fallback: display error frame continuously
#         while True:
#             frame = np.zeros((480, 640, 3), dtype=np.uint8)
#             cv2.putText(frame, "CAMERA ERROR", (50, 240), cv2.FONT_HERSHEY_SIMPLEX,
#                         1, (0, 0, 255), 2, cv2.LINE_AA)
#             ret, buffer = cv2.imencode('.jpg', frame)
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
#             time.sleep(0.1)
#         # unreachable

#     try:
#         while True:
#             success, frame = cap.read()
#             if not success:
#                 break

#             # Process frame: draw landmarks, count reps, etc.
#             processed = trainer.process_frame(frame)

#             # If an exercise is active, emit stats to clients
#             if trainer.current_exercise != ExerciseType.NONE:
#                 fb = trainer.exercises[trainer.current_exercise]
#                 stats = {
#                     "reps": fb.counter,
#                     "feedback": fb.feedback,
#                     "rate": f"{fb.rep_rate:.1f}"
#                 }
#                 socketio.emit('stats_update', stats)

#             # Encode to JPEG and stream
#             ret, buffer = cv2.imencode('.jpg', processed)
#             if not ret:
#                 continue
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
#     except Exception as e:
#         # Log and exit on unexpected errors
#         app.logger.error(f"Frame generation error: {e}")
#     finally:
#         cap.release()

# @app.route('/')
# def index():
#     """Render main interface."""
#     return render_template('index.html')

# @app.route('/video_feed')
# def video_feed():
#     """Video streaming route. Uses multipart/x-mixed-replace."""
#     return Response(
#         generate_frames(),
#         mimetype='multipart/x-mixed-replace; boundary=frame',
#         headers={
#             'Cache-Control': 'no-cache, no-store, must-revalidate',
#             'Pragma': 'no-cache',
#             'Expires': '0'
#         }
#     )

# @socketio.on('set_exercise')
# def handle_exercise_change(message):
#     """Handle exercise selection from client and notify all clients."""
#     ex = message.get('exercise', 'none')
#     mapping = {
#         'bicep': ExerciseType.BICEP_CURL,
#         'squat': ExerciseType.SQUAT,
#         'lateral': ExerciseType.LATERAL_RAISE,
#         'none': ExerciseType.NONE
#     }
#     trainer.current_exercise = mapping.get(ex, ExerciseType.NONE)
#     socketio.emit('exercise_changed', {'exercise': ex})

# if __name__ == '__main__':
#     # Run SocketIO app
#     socketio.run(app, host='127.0.0.1', port=5000, debug=True)
from flask import Flask, Response, render_template, jsonify
import cv2
import mediapipe as mp
import numpy as np
import os
import joblib
import threading
import time

app = Flask(__name__, static_url_path='/static')

# Initialize Mediapipe Pose model
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Path settings (adjust these to your environment)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(PROJECT_ROOT, 'exercise_form_model.pkl')
scaler_path = os.path.join(PROJECT_ROOT, 'scaler.pkl')

# Load the model and scaler if they exist
try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    model_loaded = True
    print("Model loaded successfully")
except:
    model_loaded = False
    print("Model not found or could not be loaded")

# Global variables for exercise tracking
counter_bicep = 0
counter_squat = 0
counter_lateral_raise = 0
stage_bicep = None
stage_squat = None
stage_lateral_raise = None
current_exercise = "None"
feedback = "Ready"
feedback_color = (0, 255, 0)  # Green
accuracy = 0
form_correct = True

# Flag to control the capture thread
camera_running = False
capture_thread = None
last_frame = None

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # Point A
    b = np.array(b)  # Point B
    c = np.array(c)  # Point C
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)  # Ensure value is between -1 and 1
    angle = np.arccos(cosine_angle)
    angle = np.degrees(angle)
    
    return angle

# Function to check if the exercise form is correct
def exercise_form_is_correct(bicep_angle, squat_angle, lateral_raise_angle):
    global current_exercise
    
    if current_exercise == "Bicep Curl":
        return 30 < bicep_angle < 160
    elif current_exercise == "Squat":
        return 90 < squat_angle < 160
    elif current_exercise == "Lateral Raise":
        return lateral_raise_angle < 90
    # Default
    return True

# Draw UI on the frame
def draw_ui(image, feedback, feedback_color, accuracy, counter_bicep, counter_squat, counter_lateral_raise, stage_bicep, stage_squat, stage_lateral_raise):
    height, width, _ = image.shape

    # Dark transparent overlay for a modern look
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 0), -1)  
    alpha = 0.3  # Transparency factor
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    # Feedback box
    cv2.rectangle(image, (50, 380), (width - 50, 430), feedback_color, -1)
    cv2.putText(image, f"Form Feedback: {feedback}", (60, 415), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Confidence Score
    cv2.putText(image, f"Confidence: {accuracy:.2f}%", (width - 250, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Exercise table
    start_x = 50
    cv2.rectangle(image, (start_x, 450), (width - 50, 500), (50, 50, 50), -1)  # Header
    cv2.putText(image, "Exercise", (start_x + 20, 485), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(image, "Reps", (start_x + 200, 485), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(image, "Stage", (start_x + 350, 485), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    exercises = [("Bicep Curl", counter_bicep, stage_bicep),
                 ("Squat", counter_squat, stage_squat),
                 ("Lateral Raise", counter_lateral_raise, stage_lateral_raise)]
    
    start_y = 510
    for ex_name, reps, stage in exercises:
        cv2.rectangle(image, (start_x, start_y), (width - 50, start_y + 40), (30, 30, 30), -1)
        cv2.putText(image, ex_name, (start_x + 20, start_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, str(reps), (start_x + 220, start_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, stage if stage else "N/A", (start_x + 370, start_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        start_y += 50  # Move to next row

    return image

def process_frame(frame, pose):
    global counter_bicep, counter_squat, counter_lateral_raise
    global stage_bicep, stage_squat, stage_lateral_raise
    global current_exercise, feedback, feedback_color, accuracy, form_correct
    
    # Convert image to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    
    # Pose detection
    results = pose.process(image)
    
    # Convert back to BGR for OpenCV rendering
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Extract landmarks and calculate accuracy
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

        # Get coordinates for lateral raise
        shoulder_left = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, 
                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        elbow_left = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, 
                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        wrist_left = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, 
                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

        # Calculate angles
        angle_bicep = calculate_angle(shoulder, elbow, wrist)
        angle_squat = calculate_angle(hip, knee, ankle)
        angle_lateral_raise = calculate_angle(shoulder_left, elbow_left, wrist_left)

        # Check form correctness based on current exercise
        form_correct = exercise_form_is_correct(angle_bicep, angle_squat, angle_lateral_raise)
        
        # Feedback text and color
        if form_correct:
            feedback = "Correct Form"
            feedback_color = (0, 255, 0)  # Green
        else:
            feedback = "Incorrect Form"
            feedback_color = (0, 0, 255)  # Red                   

        # Visualize angles
        elbow_coord = tuple(np.multiply(elbow, [frame.shape[1], frame.shape[0]]).astype(int))
        cv2.putText(image, f"Bicep: {int(angle_bicep)}°", 
                    elbow_coord, 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        knee_coord = tuple(np.multiply(knee, [frame.shape[1], frame.shape[0]]).astype(int))
        cv2.putText(image, f"Squat: {int(angle_squat)}°", 
                    knee_coord, 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        shoulder_coord = tuple(np.multiply(shoulder_left, [frame.shape[1], frame.shape[0]]).astype(int))
        cv2.putText(image, f"Lateral: {int(angle_lateral_raise)}°", 
                    shoulder_coord, 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        # Bicep curl counter logic
        if angle_bicep > 160:
            stage_bicep = "down"
        if angle_bicep < 30 and stage_bicep == "down":
            stage_bicep = "up"
            counter_bicep += 1
            current_exercise = "Bicep Curl"

        # Squat counter logic
        if angle_squat > 160:
            stage_squat = "up"
        if angle_squat < 90 and stage_squat == "up":
            stage_squat = "down"
            counter_squat += 1
            current_exercise = "Squat"

        # Lateral raise counter logic
        if angle_lateral_raise > 80:
            stage_lateral_raise = "up"
        if angle_lateral_raise < 40 and stage_lateral_raise == "up":
            stage_lateral_raise = "down"
            counter_lateral_raise += 1
            current_exercise = "Lateral Raise"

        # If model is loaded, predict form correctness
        if model_loaded:
            # Scale input data
            scaled_input = scaler.transform([[angle_bicep, angle_squat, angle_lateral_raise]])
            # Predict form correctness
            prediction = model.predict(scaled_input)[0]
            prediction_proba = model.predict_proba(scaled_input)[0]
            # Update feedback
            if prediction == 0:  # Assuming 0 is incorrect, 1 is correct
                feedback = f"Incorrect Form ({prediction_proba[0]*100:.0f}%)"
                feedback_color = (0, 0, 255)  # Red
                form_correct = False
            else:
                feedback = f"Correct Form ({prediction_proba[1]*100:.0f}%)"
                feedback_color = (0, 255, 0)  # Green
                form_correct = True

    except Exception as e:
        print(f"Error: {e}")
        accuracy = 0

    # Render pose landmarks
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
    
    # Apply UI elements
    image = draw_ui(image, feedback, feedback_color, accuracy, counter_bicep, counter_squat, counter_lateral_raise, 
                    stage_bicep, stage_squat, stage_lateral_raise)
    
    return image

def capture_camera():
    global camera_running, last_frame
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Initialize Mediapipe Pose model
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while camera_running:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process the frame
            processed_frame = process_frame(frame, pose)
            
            # Convert to JPEG for streaming
            _, buffer = cv2.imencode('.jpg', processed_frame)
            last_frame = buffer.tobytes()
            
            # Slow down to reduce CPU usage
            time.sleep(0.03)
    
    # Release resources
    cap.release()

def generate_frames():
    global last_frame
    while camera_running:
        if last_frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + last_frame + b'\r\n')
        else:
            time.sleep(0.1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_camera')
def start_camera():
    global camera_running, capture_thread
    
    if not camera_running:
        camera_running = True
        capture_thread = threading.Thread(target=capture_camera)
        capture_thread.daemon = True
        capture_thread.start()
        return jsonify({'status': 'Camera started'})
    
    return jsonify({'status': 'Camera already running'})

@app.route('/stop_camera')
def stop_camera():
    global camera_running
    
    if camera_running:
        camera_running = False
        # Give time for the thread to close
        time.sleep(1)
        return jsonify({'status': 'Camera stopped'})
    
    return jsonify({'status': 'Camera already stopped'})

@app.route('/reset_counters')
def reset_counters():
    global counter_bicep, counter_squat, counter_lateral_raise
    global stage_bicep, stage_squat, stage_lateral_raise
    
    counter_bicep = 0
    counter_squat = 0
    counter_lateral_raise = 0
    stage_bicep = None
    stage_squat = None
    stage_lateral_raise = None
    
    return jsonify({'status': 'Counters reset'})

@app.route('/get_stats')
def get_stats():
    return jsonify({
        'bicep_count': counter_bicep,
        'squat_count': counter_squat,
        'lateral_count': counter_lateral_raise,
        'current_exercise': current_exercise,
        'accuracy': accuracy,
        'feedback': feedback,
        'form_correct': form_correct
    })

if __name__ == '__main__':
    app.run(debug=True, threaded=True)