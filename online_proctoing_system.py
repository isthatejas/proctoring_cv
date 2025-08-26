import cv2
import mediapipe as mp
import numpy as np
import time
import threading
from collections import deque
import warnings
warnings.filterwarnings("ignore")

class OnlineProctoringSystem:
    def __init__(self):
        # Initialize MediaPipe components
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Initialize face detection and mesh
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.7
        )
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=3,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )

        # Alert thresholds and timers
        self.GAZE_THRESHOLD_TIME = 25  # seconds
        self.BRIGHTNESS_THRESHOLD = 45  # minimum brightness level
        self.MAX_FACES = 1  # maximum allowed faces

        # Tracking variables
        self.gaze_start_time = {}
        self.alert_log = []
        self.frame_count = 0
        self.fps = 0

        # Eye landmarks for gaze tracking (MediaPipe indices)
        self.LEFT_EYE_LANDMARKS = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.RIGHT_EYE_LANDMARKS = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

        # Direction tracking
        self.direction_history = deque(maxlen=15)

    def calculate_brightness(self, frame):
        """Calculate average brightness of the frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return np.mean(gray)

    def check_illumination(self, frame):
        """Check if illumination is adequate"""
        brightness = self.calculate_brightness(frame)
        if brightness < self.BRIGHTNESS_THRESHOLD:
            self.add_alert("Poor illumination detected")
            return False, brightness
        return True, brightness

    def detect_multiple_faces(self, frame):
        """Detect and count faces in the frame"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)

        face_count = 0
        if results.detections:
            face_count = len(results.detections)

        if face_count == 0:
            self.add_alert("No face detected")
            return face_count, None
        elif face_count > self.MAX_FACES:
            self.add_alert(f"Multiple faces detected: {face_count}")
            return face_count, results.detections

        return face_count, results.detections

    def get_gaze_direction(self, landmarks, frame_shape):
        """Estimate gaze direction based on eye landmarks"""
        h, w = frame_shape[:2]

        # Get left and right eye centers
        left_eye_center = self.get_eye_center(landmarks, self.LEFT_EYE_LANDMARKS, w, h)
        right_eye_center = self.get_eye_center(landmarks, self.RIGHT_EYE_LANDMARKS, w, h)

        if left_eye_center is None or right_eye_center is None:
            return "center"

        # Get nose tip for reference
        nose_tip = landmarks[1]  # Nose tip landmark
        nose_x = int(nose_tip.x * w)

        # Calculate eye positions relative to nose
        left_eye_x = left_eye_center[0]
        right_eye_x = right_eye_center[0]

        avg_eye_x = (left_eye_x + right_eye_x) / 2

        # Determine gaze direction
        offset = avg_eye_x - nose_x

        if offset > 20:
            return "right"
        elif offset < -20:
            return "left"
        else:
            return "center"

    def get_eye_center(self, landmarks, eye_landmarks, w, h):
        """Calculate center of eye based on landmarks"""
        eye_points = []
        for idx in eye_landmarks:
            if idx < len(landmarks):
                point = landmarks[idx]
                eye_points.append((int(point.x * w), int(point.y * h)))

        if not eye_points:
            return None

        # Calculate center
        x = sum([p[0] for p in eye_points]) // len(eye_points)
        y = sum([p[1] for p in eye_points]) // len(eye_points)
        return (x, y)

    def track_gaze_duration(self, direction):
        """Track how long person has been looking in a direction"""
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
                    self.add_alert(f"Looking {direction} for {duration:.1f} seconds")
                    self.gaze_start_time[direction] = current_time  # Reset timer
        else:
            # Reset all timers if looking center
            self.gaze_start_time.clear()

    def analyze_eye_tracking(self, frame):
        """Perform eye tracking and gaze analysis"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Get gaze direction
                direction = self.get_gaze_direction(face_landmarks.landmark, frame.shape)

                # Track gaze duration
                self.track_gaze_duration(direction)

                # Draw eye landmarks
                self.draw_eye_landmarks(frame, face_landmarks.landmark, frame.shape)

                # Add direction text
                cv2.putText(frame, f"Gaze: {direction}", (50, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                return direction

        return "center"

    def draw_eye_landmarks(self, frame, landmarks, frame_shape):
        """Draw eye landmarks on the frame"""
        h, w = frame_shape[:2]

        # Draw left eye
        left_eye_points = []
        for idx in self.LEFT_EYE_LANDMARKS:
            if idx < len(landmarks):
                point = landmarks[idx]
                x, y = int(point.x * w), int(point.y * h)
                left_eye_points.append((x, y))
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

        # Draw right eye
        right_eye_points = []
        for idx in self.RIGHT_EYE_LANDMARKS:
            if idx < len(landmarks):
                point = landmarks[idx]
                x, y = int(point.x * w), int(point.y * h)
                right_eye_points.append((x, y))
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    def add_alert(self, message):
        """Add alert to log with timestamp"""
        timestamp = time.strftime("%H:%M:%S")
        alert = f"[{timestamp}] {message}"
        self.alert_log.append(alert)
        print(f"ALERT: {alert}")

    def draw_alerts(self, frame):
        """Draw recent alerts on frame"""
        y_offset = 30
        recent_alerts = self.alert_log[-5:]  # Show last 5 alerts

        for alert in recent_alerts:
            cv2.putText(frame, alert, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            y_offset += 20

    def process_frame(self, frame):
        """Main processing function for each frame"""
        self.frame_count += 1

        # 1. Check illumination
        illumination_ok, brightness = self.check_illumination(frame)

        # 2. Detect multiple faces
        face_count, detections = self.detect_multiple_faces(frame)

        # 3. Analyze eye tracking (only if exactly one face)
        gaze_direction = "center"
        if face_count == 1:
            gaze_direction = self.analyze_eye_tracking(frame)

        # 4. Draw face detection boxes
        if detections:
            for detection in detections:
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)

                # Draw bounding box
                color = (0, 255, 0) if face_count == 1 else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)

                # Draw confidence score
                confidence = detection.score[0]
                cv2.putText(frame, f"Confidence: {confidence:.2f}", 
                           (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # 5. Draw information on frame
        cv2.putText(frame, f"Faces: {face_count}", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f"Brightness: {brightness:.1f}", (50, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (50, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # 6. Draw alerts
        self.draw_alerts(frame)

        return frame

    def run_proctoring(self, video_source=0):
        """Main function to run the proctoring system"""
        cap = cv2.VideoCapture(video_source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # FPS calculation
        fps_counter = 0
        fps_timer = time.time()

        print("Starting Online Proctoring System...")
        print("Press 'q' to quit, 's' to save current frame")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break

            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)

            # Process the frame
            processed_frame = self.process_frame(frame.copy())

            # Calculate FPS
            fps_counter += 1
            if time.time() - fps_timer >= 1.0:
                self.fps = fps_counter
                fps_counter = 0
                fps_timer = time.time()

            # Display the frame
            cv2.imshow('Online Proctoring System', processed_frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"proctoring_frame_{int(time.time())}.jpg"
                cv2.imwrite(filename, processed_frame)
                print(f"Frame saved as {filename}")

        # Cleanup
        cap.release()
        cv2.destroyAllWindows()

        # Print summary
        print(f"\nSession Summary:")
        print(f"Total frames processed: {self.frame_count}")
        print(f"Total alerts: {len(self.alert_log)}")
        print("\nAlert Log:")
        for alert in self.alert_log:
            print(alert)

# Usage example
if __name__ == "__main__":
    proctoring_system = OnlineProctoringSystem()
    proctoring_system.run_proctoring()