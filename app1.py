import cv2
import numpy as np
import tensorflow as tf
import pygame
from flask import Flask, Response, render_template
import time

# Initialize Flask
app = Flask(__name__)

# Initialize Pygame
pygame.init()
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
skeleton_surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)

# Load Skeleton Image (Ensure you have a realistic skeleton image, transparent background)
skeleton_image = pygame.image.load(r'E:\job related docs\py development\—Pngtree—human body skeleton model_4453747.png')  # Correct path
skeleton_image = pygame.transform.scale(skeleton_image, (WINDOW_WIDTH, WINDOW_HEIGHT))

# Initialize the Pose Detection Model
# Ensure eager execution
tf.compat.v1.enable_eager_execution()

class PoseDetector:
    def __init__(self):
        print("Loading MoveNet model...")
        try:
            model_path = 'C:/Users/SURABHI/Downloads/movenet-tensorflow2-singlepose-lightning-v4'  # Correct path
            self.model = tf.saved_model.load(model_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
        self.input_size = 192
        print("Model loaded successfully!")

    def detect_pose(self, frame):
        if self.model is None:
            return {}

        img = tf.image.resize_with_pad(tf.expand_dims(frame, axis=0), self.input_size, self.input_size)
        img = tf.cast(img, dtype=tf.int32)

        outputs = self.model.signatures["serving_default"](img)
        
        # Ensure eager execution and evaluate tensor
        keypoints = outputs['output_0'].numpy()[0][0]

        height, width = frame.shape[:2]
        keypoints_dict = {}

        min_confidence = 0.2
        for idx, kp in enumerate(keypoints):
            y, x, score = kp
            if score > min_confidence:
                x_px = min(max(0, int(x * width)), width - 1)
                y_px = min(max(0, int(y * height)), height - 1)
                keypoints_dict[idx] = (x_px, y_px)

        return keypoints_dict

pose_detector = PoseDetector()

# Skeleton Animator to display realistic skeleton
class SkeletonAnimator:
    def __init__(self):
        self.joints = {}
        self.bones = [
            (0, 5), (0, 6), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),
            (5, 11), (6, 12), (11, 12), (11, 13), (12, 14), (13, 15), (14, 16)
        ]
        self.joint_colors = {
            0: (255, 0, 0),    # Nose
            5: (0, 255, 0),    # Left shoulder
            6: (0, 255, 0),    # Right shoulder
            7: (0, 255, 255),  # Left elbow
            8: (0, 255, 255),  # Right elbow
            9: (255, 255, 0),  # Left wrist
            10: (255, 255, 0), # Right wrist
            11: (255, 0, 255), # Left hip
            12: (255, 0, 255), # Right hip
            13: (0, 0, 255),   # Left knee
            14: (0, 0, 255),   # Right knee
            15: (128, 0, 255), # Left ankle
            16: (128, 0, 255), # Right ankle
        }
        self.sitting = False

    def update_positions(self, keypoints):
        self.joints = keypoints

        # Check if the person is sitting (example logic based on joint positions)
        if keypoints.get(11) and keypoints.get(12):  # Left and right hips
            left_hip = keypoints[11]
            right_hip = keypoints[12]
            if left_hip[1] < 0.5 and right_hip[1] < 0.5:
                self.sitting = True
            else:
                self.sitting = False

    def draw(self, surface):
        surface.fill((0, 0, 0, 0))  # Clear surface

        # Exaggerated funny animation when sitting or raising hands
        for start, end in self.bones:
            if start in self.joints and end in self.joints:
                start_pos = self.joints[start]
                end_pos = self.joints[end]
                if self.sitting:
                    start_pos = (start_pos[0], start_pos[1] + 20)  # Sitting pose (lower)
                    end_pos = (end_pos[0], end_pos[1] + 20)  # Sitting pose (lower)
                pygame.draw.line(surface, (255, 255, 255), start_pos, end_pos, 5)  # White lines for bones

        # Draw joints as circles (using pre-defined colors)
        for joint_id, joint_pos in self.joints.items():
            if joint_id in self.joint_colors:
                color = self.joint_colors[joint_id]
                pygame.draw.circle(surface, color, joint_pos, 10)  # Draw joint as a circle
                if joint_id == 5:  # Left shoulder, exaggerated animation when hands raised
                    pygame.draw.circle(surface, (255, 255, 0), (joint_pos[0], joint_pos[1] - 20), 10)  # Raise hands funny effect

def generate_frames():
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Error: Could not initialize camera.")
        return

    while True:
        success, frame = camera.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        keypoints = pose_detector.detect_pose(frame)

        # Debugging: Print keypoints to check if pose detection is working
        print(keypoints)

        skeleton_animator.update_positions(keypoints)
        skeleton_animator.draw(skeleton_surface)

        # Convert Pygame surface to OpenCV format
        skeleton_image = pygame.surfarray.array3d(skeleton_surface)
        skeleton_image = np.transpose(skeleton_image, (1, 0, 2))
        skeleton_image = cv2.resize(skeleton_image, (frame.shape[1], frame.shape[0]))

        # Superimpose the skeleton image onto the webcam frame
        combined_frame = cv2.addWeighted(frame, 0.7, skeleton_image, 0.3, 0)

        ret, buffer = cv2.imencode('.jpg', combined_frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    camera.release()

skeleton_animator = SkeletonAnimator()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
