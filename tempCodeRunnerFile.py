from flask import Flask, render_template, Response
from flask_socketio import SocketIO
from pose_model import FitnessTrainer, ExerciseType
import cv2
import numpy as np

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")
trainer = FitnessTrainer()

def generate_frames():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        while True:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "CAMERA ERROR", (50, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.1)
        return

    while True:
        try:
            success, frame = cap.read()
            if not success:
                break
            
            frame = trainer.process_frame(frame)
            
            if trainer.current_exercise != ExerciseType.NONE:
                current_fb = trainer.exercises[trainer.current_exercise]
                stats = {
                    "reps": current_fb.counter,
                    "feedback": current_fb.feedback,
                    "rate": f"{current_fb.rep_rate:.1f}"
                }
                socketio.emit('stats_update', stats)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        except Exception as e:
            print(f"Camera Error: {str(e)}")
            break
    
    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame',
        headers={
            'Cache-Control': 'no-cache, no-store, must-revalidate',
            'Pragma': 'no-cache',
            'Expires': '0'
        }
    )

@socketio.on('set_exercise')
def handle_exercise_change(exercise):
    exercise_map = {
        'bicep': ExerciseType.BICEP_CURL,
        'squat': ExerciseType.SQUAT,
        'lateral': ExerciseType.LATERAL_RAISE,
        'none': ExerciseType.NONE
    }
    trainer.current_exercise = exercise_map.get(exercise, ExerciseType.NONE)
    socketio.emit('exercise_changed', {'exercise': exercise})

if __name__ == '__main__':
    socketio.run(app, host='127.0.0.1', port=5000, debug=True)