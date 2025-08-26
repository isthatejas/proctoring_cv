
from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO, emit
import cv2
import mediapipe as mp
import numpy as np
import time
import threading
from collections import deque
import base64
import json
import math
from datetime import datetime

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*")

class WebProctoring:
    def __init__(self):
        # Initialize MediaPipe components
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils

        # Initialize detection models
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.7
        )
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=3,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )

        # Thresholds and settings
        self.GAZE_THRESHOLD_TIME = 15
        self.BRIGHTNESS_THRESHOLD = 80
        self.MAX_FACES = 1

        # Tracking variables
        self.gaze_start_time = {}
        self.alert_log = []
        self.frame_count = 0
        self.fps = 0
        self.session_start_time = time.time()
        self.is_recording = False

        # Advanced features
        self.blink_threshold = 0.25
        self.blink_counter = 0
        self.hand_near_face_counter = 0
        self.suspicious_activity_score = 0
        self.activity_history = deque(maxlen=300)

        # Eye landmarks for tracking
        self.LEFT_EYE_LANDMARKS = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.RIGHT_EYE_LANDMARKS = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

        # Statistics
        self.stats = {
            'session_duration': 0,
            'total_alerts': 0,
            'face_detections': 0,
            'brightness_avg': 0,
            'integrity_score': 100
        }

    def calculate_brightness(self, frame):
        """Calculate average brightness of the frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return np.mean(gray)

    def detect_faces(self, frame):
        """Detect faces and return count with bounding boxes"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)

        face_count = 0
        detections = []

        if results.detections:
            face_count = len(results.detections)
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                confidence = detection.score[0]

                detections.append({
                    'x': x, 'y': y, 'width': width, 'height': height,
                    'confidence': confidence
                })

        return face_count, detections

    def get_gaze_direction(self, landmarks, frame_shape):
        """Estimate gaze direction"""
        if len(landmarks) < 468:
            return "center"

        h, w = frame_shape[:2]

        # Get eye centers
        left_eye_center = self.get_eye_center(landmarks, self.LEFT_EYE_LANDMARKS, w, h)
        right_eye_center = self.get_eye_center(landmarks, self.RIGHT_EYE_LANDMARKS, w, h)

        if left_eye_center is None or right_eye_center is None:
            return "center"

        # Get nose tip for reference
        nose_tip = landmarks[1]
        nose_x = int(nose_tip.x * w)

        # Calculate average eye position
        avg_eye_x = (left_eye_center[0] + right_eye_center[0]) / 2
        offset = avg_eye_x - nose_x

        if offset > 20:
            return "right"
        elif offset < -20:
            return "left"
        else:
            return "center"

    def get_eye_center(self, landmarks, eye_landmarks, w, h):
        """Calculate center of eye"""
        eye_points = []
        for idx in eye_landmarks[:6]:  # Use first 6 landmarks
            if idx < len(landmarks):
                point = landmarks[idx]
                eye_points.append((int(point.x * w), int(point.y * h)))

        if len(eye_points) < 3:
            return None

        x = sum([p[0] for p in eye_points]) // len(eye_points)
        y = sum([p[1] for p in eye_points]) // len(eye_points)
        return (x, y)

    def track_gaze_duration(self, direction):
        """Track gaze duration and emit alerts"""
        current_time = time.time()

        # Clean up old directions
        for dir_key in list(self.gaze_start_time.keys()):
            if dir_key != direction:
                self.gaze_start_time.pop(dir_key, None)

        # Track current direction
        if direction not in ["center"]:
            if direction not in self.gaze_start_time:
                self.gaze_start_time[direction] = current_time
            else:
                duration = current_time - self.gaze_start_time[direction]
                if duration >= self.GAZE_THRESHOLD_TIME:
                    self.emit_alert(f"Looking {direction} for {duration:.1f} seconds", "warning")
                    self.gaze_start_time[direction] = current_time
        else:
            self.gaze_start_time.clear()

    def detect_hands_near_face(self, frame):
        """Detect hands near face area"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        hands_detected = False
        hand_positions = []

        if results.multi_hand_landmarks:
            h, w, _ = frame.shape

            for hand_landmarks in results.multi_hand_landmarks:
                # Calculate hand center
                hand_points = [(landmark.x * w, landmark.y * h) 
                              for landmark in hand_landmarks.landmark]

                hand_center_x = sum([p[0] for p in hand_points]) / len(hand_points)
                hand_center_y = sum([p[1] for p in hand_points]) / len(hand_points)

                # Check if hand is in upper portion (face area)
                if hand_center_y < h * 0.6:
                    hands_detected = True
                    hand_positions.append((int(hand_center_x), int(hand_center_y)))

                    # Draw hand landmarks
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                    )

        if hands_detected:
            self.hand_near_face_counter += 1
            if self.hand_near_face_counter > 30:  # 1 second at 30fps
                self.emit_alert("Hand detected near face area", "warning")
                self.hand_near_face_counter = 0
        else:
            self.hand_near_face_counter = max(0, self.hand_near_face_counter - 1)

        return hands_detected, hand_positions

    def analyze_eye_tracking(self, frame):
        """Perform comprehensive eye and face analysis"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        gaze_direction = "center"

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Get gaze direction
                gaze_direction = self.get_gaze_direction(face_landmarks.landmark, frame.shape)

                # Track gaze duration
                self.track_gaze_duration(gaze_direction)

                # Draw eye landmarks
                self.draw_eye_landmarks(frame, face_landmarks.landmark, frame.shape)

                break  # Process only first face

        return gaze_direction

    def draw_eye_landmarks(self, frame, landmarks, frame_shape):
        """Draw eye landmarks on frame"""
        h, w = frame_shape[:2]

        # Draw eye points
        for idx in self.LEFT_EYE_LANDMARKS[:8] + self.RIGHT_EYE_LANDMARKS[:8]:
            if idx < len(landmarks):
                point = landmarks[idx]
                x, y = int(point.x * w), int(point.y * h)
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

    def emit_alert(self, message, alert_type="info"):
        """Emit alert to web interface"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        alert_data = {
            'message': message,
            'type': alert_type,
            'timestamp': timestamp
        }

        self.alert_log.append(alert_data)
        self.stats['total_alerts'] += 1

        # Emit to web interface
        socketio.emit('new_alert', alert_data)

        print(f"ALERT [{alert_type.upper()}]: {message}")

    def calculate_integrity_score(self):
        """Calculate current integrity score"""
        base_score = 100

        # Deduct points for different alert types
        for alert in self.alert_log[-10:]:  # Last 10 alerts
            if alert['type'] == 'critical':
                base_score -= 15
            elif alert['type'] == 'warning':
                base_score -= 8
            elif alert['type'] == 'info':
                base_score -= 3

        return max(0, min(100, base_score))

    def process_frame(self, frame):
        """Main frame processing function"""
        self.frame_count += 1

        # 1. Check brightness
        brightness = self.calculate_brightness(frame)
        if brightness < self.BRIGHTNESS_THRESHOLD:
            if self.frame_count % 30 == 0:  # Alert every second
                self.emit_alert("Improper illumination. Please ensure your face is well-lit.", "warning")

        # 2. Detect faces
        face_count, detections = self.detect_faces(frame)

        if face_count == 0:
            if self.frame_count % 30 == 0:
                self.emit_alert("No face detected. Please position yourself in front of the camera.", "warning")
        elif face_count > self.MAX_FACES:
            self.emit_alert(f"Multiple faces detected ({face_count}). Only one person allowed.", "critical")

        # 3. Detect hands (if single face)
        hands_detected = False
        if face_count == 1:
            hands_detected, _ = self.detect_hands_near_face(frame)
            gaze_direction = self.analyze_eye_tracking(frame)

        # 4. Draw face detection boxes
        for detection in detections:
            color = (0, 255, 0) if face_count == 1 else (0, 0, 255)
            cv2.rectangle(frame, 
                         (detection['x'], detection['y']), 
                         (detection['x'] + detection['width'], detection['y'] + detection['height']), 
                         color, 2)

            # Draw confidence
            cv2.putText(frame, f"Conf: {detection['confidence']:.2f}", 
                       (detection['x'], detection['y'] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # 5. Draw status information
        info_y = 30
        cv2.putText(frame, f"Faces: {face_count}", (10, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        info_y += 30
        cv2.putText(frame, f"Brightness: {brightness:.0f}", (10, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        info_y += 30
        cv2.putText(frame, f"Hands: {'Yes' if hands_detected else 'No'}", (10, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # 6. Update statistics
        self.stats['brightness_avg'] = brightness
        self.stats['face_detections'] = face_count
        self.stats['session_duration'] = time.time() - self.session_start_time
        self.stats['integrity_score'] = self.calculate_integrity_score()

        return frame

# Global proctoring instance
proctoring_system = WebProctoring()
camera = None

def get_camera():
    """Get camera instance"""
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        camera.set(cv2.CAP_PROP_FPS, 30)
    return camera

def generate_frames():
    """Generate video frames for streaming"""
    global proctoring_system

    camera = get_camera()
    fps_counter = 0
    fps_timer = time.time()

    while True:
        success, frame = camera.read()
        if not success:
            break

        # Flip frame for mirror effect
        frame = cv2.flip(frame, 1)

        # Process frame through proctoring system
        if proctoring_system.is_recording:
            processed_frame = proctoring_system.process_frame(frame.copy())
        else:
            processed_frame = frame

        # Calculate FPS
        fps_counter += 1
        if time.time() - fps_timer >= 1.0:
            proctoring_system.fps = fps_counter
            fps_counter = 0
            fps_timer = time.time()

        # Encode frame
        ret, buffer = cv2.imencode('.jpg', processed_frame, 
                                  [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_proctoring', methods=['POST'])
def start_proctoring():
    """Start proctoring session"""
    global proctoring_system
    proctoring_system.is_recording = True
    proctoring_system.session_start_time = time.time()
    proctoring_system.alert_log = []
    proctoring_system.frame_count = 0

    socketio.emit('proctoring_started', {'status': 'started'})
    return jsonify({'status': 'started', 'message': 'Proctoring session started'})

@app.route('/stop_proctoring', methods=['POST'])
def stop_proctoring():
    """Stop proctoring session"""
    global proctoring_system
    proctoring_system.is_recording = False

    # Generate session report
    report = {
        'session_duration': time.time() - proctoring_system.session_start_time,
        'total_alerts': len(proctoring_system.alert_log),
        'integrity_score': proctoring_system.stats['integrity_score'],
        'alerts': proctoring_system.alert_log[-10:]  # Last 10 alerts
    }

    socketio.emit('proctoring_stopped', report)
    return jsonify({'status': 'stopped', 'report': report})

@app.route('/get_stats')
def get_stats():
    """Get current session statistics"""
    global proctoring_system
    stats = proctoring_system.stats.copy()
    stats['session_duration'] = time.time() - proctoring_system.session_start_time
    stats['is_recording'] = proctoring_system.is_recording
    stats['fps'] = proctoring_system.fps
    return jsonify(stats)

@app.route('/get_alerts')
def get_alerts():
    """Get recent alerts"""
    global proctoring_system
    return jsonify({
        'alerts': proctoring_system.alert_log[-10:],  # Last 10 alerts
        'total_count': len(proctoring_system.alert_log)
    })

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print('Client connected')
    emit('connected', {'data': 'Connected to proctoring system'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print('Client disconnected')

if __name__ == '__main__':
    print("Starting AI Proctoring Web Application...")
    print("Access the application at: http://localhost:5000")
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)