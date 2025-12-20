import numpy as np
import cv2
import mediapipe as mp
import time
from enum import Enum

class ExerciseType(Enum):
    BICEP_CURL = 1
    SQUAT = 2
    LATERAL_RAISE = 3
    NONE = 4

class ExerciseStage(Enum):
    DOWN = 1
    UP = 2
    NONE = 3

class ExerciseFeedback:
    def __init__(self):
        self.counter = 0
        self.stage = ExerciseStage.NONE
        self.feedback = "Waiting..."
        self.is_correct_form = False
        self.angle = 0
        self.last_rep_time = time.time()
        self.rep_rate = 0

class FitnessTrainer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            model_complexity=0  # Lower complexity for faster processing
        )
        self.current_exercise = ExerciseType.NONE
        self.exercises = {
            ExerciseType.BICEP_CURL: ExerciseFeedback(),
            ExerciseType.SQUAT: ExerciseFeedback(),
            ExerciseType.LATERAL_RAISE: ExerciseFeedback()
        }
        self.mp_drawing = mp.solutions.drawing_utils
        self.ui_font = cv2.FONT_HERSHEY_SIMPLEX

    def calculate_angle(self, a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        return np.degrees(np.arccos(cosine_angle))

    def update_bicep_curl(self, landmarks, feedback):
        try:
            shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                       landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                    landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                    landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            feedback.angle = self.calculate_angle(shoulder, elbow, wrist)
            feedback.is_correct_form = (30 < feedback.angle < 160)
            feedback.feedback = "Correct Form" if feedback.is_correct_form else "Keep elbow stable"

            if feedback.angle > 160:
                feedback.stage = ExerciseStage.DOWN
            if feedback.angle < 30 and feedback.stage == ExerciseStage.DOWN:
                feedback.stage = ExerciseStage.UP
                self._increment_counter(feedback)
        except Exception as e:
            feedback.feedback = f"Error: {str(e)}"

    def update_squat(self, landmarks, feedback):
        try:
            hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x,
                  landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee = [landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                   landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            ankle = [landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                    landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            feedback.angle = self.calculate_angle(hip, knee, ankle)
            feedback.is_correct_form = (70 < feedback.angle < 170)
            feedback.feedback = "Correct Form" if feedback.is_correct_form else "Keep back straight"

            if feedback.angle > 170:
                feedback.stage = ExerciseStage.UP
            if feedback.angle < 70 and feedback.stage == ExerciseStage.UP:
                feedback.stage = ExerciseStage.DOWN
                self._increment_counter(feedback)
        except Exception as e:
            feedback.feedback = f"Error: {str(e)}"

    def update_lateral_raise(self, landmarks, feedback):
        try:
            shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                       landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                    landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                    landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            feedback.angle = self.calculate_angle(shoulder, elbow, wrist)
            wrist_height_ratio = shoulder[1] - wrist[1]
            elbow_height_ratio = shoulder[1] - elbow[1]

            feedback.is_correct_form = (160 < feedback.angle < 200) and (wrist_height_ratio > 0)
            feedback.feedback = "Correct Form" if feedback.is_correct_form else "Raise arms sideways"

            if wrist_height_ratio > 0.15 * elbow_height_ratio:
                feedback.stage = ExerciseStage.UP
            elif wrist_height_ratio < 0.05 * elbow_height_ratio and feedback.stage == ExerciseStage.UP:
                feedback.stage = ExerciseStage.DOWN
                self._increment_counter(feedback)
        except Exception as e:
            feedback.feedback = f"Error: {str(e)}"

    def _increment_counter(self, feedback):
        feedback.counter += 1
        now = time.time()
        feedback.rep_rate = 60 / (now - feedback.last_rep_time) if (now - feedback.last_rep_time) > 0 else 0
        feedback.last_rep_time = now

    def process_frame(self, image):
        try:
            start_time = time.time()  # Start time for processing the frame

            # Convert image to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            
            # Process with Mediapipe
            results = self.pose.process(image_rgb)
            
            # Convert back to BGR
            image_rgb.flags.writeable = True
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            # Process landmarks if detected
            if results.pose_landmarks:
                if self.current_exercise == ExerciseType.BICEP_CURL:
                    self.update_bicep_curl(results.pose_landmarks.landmark, 
                                          self.exercises[ExerciseType.BICEP_CURL])
                elif self.current_exercise == ExerciseType.SQUAT:
                    self.update_squat(results.pose_landmarks.landmark,
                                     self.exercises[ExerciseType.SQUAT])
                elif self.current_exercise == ExerciseType.LATERAL_RAISE:
                    self.update_lateral_raise(results.pose_landmarks.landmark,
                                             self.exercises[ExerciseType.LATERAL_RAISE])

                # Draw landmarks
                self.mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                )

            # Log the time taken for the frame processing
            end_time = time.time()
            print(f"Frame processing time: {end_time - start_time:.4f} seconds")

            return self.draw_ui(image, results.pose_landmarks if results.pose_landmarks else None)
            
        except Exception as e:
            print(f"Processing error: {str(e)}")
            return image

    def draw_ui(self, image, landmarks):
        try:
            height, width = image.shape[:2]
            
            # Dark semi-transparent overlay
            overlay = image.copy()
            cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.2, image, 0.8, 0, image)

            # Current exercise info
            current_fb = self.exercises.get(self.current_exercise, ExerciseFeedback())
            fb_color = (0, 255, 0) if current_fb.is_correct_form else (0, 0, 255)

            # Header
            cv2.rectangle(image, (0, 0), (width, 60), (50, 50, 50), -1)
            exercise_name = self.current_exercise.name.replace('_', ' ') if self.current_exercise != ExerciseType.NONE else "None"
            cv2.putText(image, f"Exercise: {exercise_name}", 
                       (20, 35), self.ui_font, 0.8, (255, 255, 255), 2)

            # Exercise stats
            start_y = 70
            cv2.rectangle(image, (20, start_y), (width - 20, start_y + 160), (50, 50, 50), -1)
            
            # Column headers
            headers = ["Exercise", "Reps", "Form", "Rate"]
            for i, header in enumerate(headers):
                cv2.putText(image, header, (20 + i*200, start_y + 30), 
                           self.ui_font, 0.7, (255, 255, 255), 2)

            # Exercise rows
            for i, (ex_type, fb) in enumerate(self.exercises.items()):
                row_y = start_y + 60 + i*40
                # Exercise name
                cv2.putText(image, ex_type.name.replace('_', ' '), (20, row_y), 
                           self.ui_font, 0.6, (255, 255, 255), 1)
                # Reps
                cv2.putText(image, str(fb.counter), (220, row_y), 
                           self.ui_font, 0.6, (255, 255, 255), 1)
                # Form status
                form_text = fb.feedback
                cv2.putText(image, form_text, (420, row_y), 
                           self.ui_font, 0.6, fb_color, 1)
                # Rep rate
                cv2.putText(image, f"{fb.rep_rate:.2f} reps/min", (620, row_y),
                           self.ui_font, 0.6, (255, 255, 255), 1)

            return image
        except Exception as e:
            print(f"UI Drawing Error: {str(e)}")
            return image

    def set_exercise(self, exercise_type):
        self.current_exercise = exercise_type

    def reset_exercise(self):
        self.current_exercise = ExerciseType.NONE
        for ex_type in self.exercises:
            self.exercises[ex_type] = ExerciseFeedback()

# Example usage for testing:
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    trainer = FitnessTrainer()

    # Set to track bicep curl for testing
    trainer.set_exercise(ExerciseType.BICEP_CURL)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame
        frame = trainer.process_frame(frame)

        # Display the resulting frame
        cv2.imshow('Fitness Trainer', frame)

        # Stop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
