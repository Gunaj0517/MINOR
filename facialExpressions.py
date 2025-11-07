import cv2
import mediapipe as mp
import numpy as np
import os
import urllib.request
import time
from collections import deque # NEW: Robustness ke liye import kiya

# --- MediaPipe Blendshapes Setup ---
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Blendshape model file download
MODEL_FILE = 'face_landmarker.task'
if not os.path.exists(MODEL_FILE):
    print(f"Downloading MediaPipe blendshapes model to {MODEL_FILE}...")
    try:
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        urllib.request.urlretrieve(url, MODEL_FILE)
        print("Download complete.")
    except Exception as e:
        print(f"Error downloading model: {e}")
        print("Please download the file manually from the URL above and place it in the same folder.")
        exit()

# MediaPipe FaceLandmarker options
options = mp.tasks.vision.FaceLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_FILE),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    output_face_blendshapes=True, # Humein blendshapes chahiye
    num_faces=1
)

# --- Robust Emotion Classifier (UPDATED) ---

# NEW: Rolling average ke liye Deque (taaki robust ho)
HISTORY_LENGTH = 10 # Pichele 10 frames ka average
blendshape_history = deque(maxlen=HISTORY_LENGTH)

# NEW: Key blendshapes jinko hum dashboard par dikhayenge
KEY_BLENDSHAPES = [
    'mouthSmileLeft', 'mouthSmileRight',
    'jawOpen',
    'eyeWideLeft', 'eyeWideRight',
    'browDownLeft', 'browDownRight',
    'mouthFrownLeft', 'mouthFrownRight',
    'eyeSquintLeft', 'eyeSquintRight'
]

# UPDATED: Function ab averaged scores par kaam karega
def classify_emotion_from_blendshapes(avg_scores):
    """
    Yeh *averaged* blendshape scores ko simple emotions mein badalta hai.
    """
    if not avg_scores:
        return "---"

    # 1. Happy (Smile + Eye Squint)
    avg_smile = (avg_scores.get('mouthSmileLeft', 0) + avg_scores.get('mouthSmileRight', 0)) / 2
    avg_squint = (avg_scores.get('eyeSquintLeft', 0) + avg_scores.get('eyeSquintRight', 0)) / 2
    if avg_smile > 0.25 and avg_squint > 0.1: # Thresholds ko robust banaya
        return "Happy"

    # 2. Surprised (Jaw Open + Eyes Wide)
    avg_jaw_open = avg_scores.get('jawOpen', 0)
    avg_eye_wide = (avg_scores.get('eyeWideLeft', 0) + avg_scores.get('eyeWideRight', 0)) / 2
    if avg_jaw_open > 0.3 and avg_eye_wide > 0.25:
        return "Surprised"

    # 3. Angry (Brows Down)
    avg_brow_down = (avg_scores.get('browDownLeft', 0) + avg_scores.get('browDownRight', 0)) / 2
    if avg_brow_down > 0.35:
        return "Angry"
        
    # 4. Sad (Mouth Frown)
    avg_frown = (avg_scores.get('mouthFrownLeft', 0) + avg_scores.get('mouthFrownRight', 0)) / 2
    if avg_frown > 0.2:
        return "Sad"

    return "Neutral"

# NEW: Dashboard (Ratio Table) draw karne ka function
def draw_dashboard(frame, avg_scores):
    """
    Screen ke right side par blendshape scores ka dashboard banata hai.
    """
    h, w, _ = frame.shape
    dashboard_width = 300
    overlay = frame.copy()
    cv2.rectangle(overlay, (w - dashboard_width, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame) # Transparency

    cv2.putText(frame, "Blendshape Ratios", (w - dashboard_width + 20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    y_pos = 80
    if not avg_scores:
        cv2.putText(frame, "Detecting...", (w - dashboard_width + 20, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        return

    for name in KEY_BLENDSHAPES:
        score = avg_scores.get(name, 0)
        score_text = f"{name}: {score:.2f}"
        cv2.putText(frame, score_text, (w - dashboard_width + 20, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Confidence bar
        bar_width = int(score * (dashboard_width - 60))
        bar_color = (0, 255, 0) # Green
        
        cv2.rectangle(frame, (w - dashboard_width + 20, y_pos + 10), 
                      (w - dashboard_width + 20 + bar_width, y_pos + 20), 
                      bar_color, -1)
        y_pos += 35

# --- Main Video Loop ---
print("Initializing MediaPipe Emotion Detection (Blendshapes)...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Unable to open the Webcam.")
    exit()

# FaceLandmarker ko create karte hain
with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Frame ko resize karte hain taaki dashboard ke liye jagah ban sake
        frame = cv2.resize(frame, (1280, 720)) # (Width, Height)
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_timestamp_ms = int(time.time() * 1000)
        
        landmarker_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
        
        current_scores = {}
        if landmarker_result.face_blendshapes:
            # Result se blendshapes nikaalte hain
            blendshape_data = landmarker_result.face_blendshapes[0]
            current_scores = {shape.category_name: shape.score for shape in blendshape_data}

        # History mein add karte hain (NEW)
        blendshape_history.append(current_scores)
        
        # --- Averaged Scores Calculate karna (NEW) ---
        avg_scores = {}
        if blendshape_history:
            # Saare scores ko sum karte hain
            for scores_dict in blendshape_history:
                for name, score in scores_dict.items():
                    avg_scores[name] = avg_scores.get(name, 0) + score
            # Average nikaalte hain
            num_frames = len(blendshape_history)
            for name in avg_scores:
                avg_scores[name] /= num_frames
        
        # Average scores se emotion classify karte hain (UPDATED)
        emotion = classify_emotion_from_blendshapes(avg_scores)
        
        # --- Dashboard Draw karna (NEW) ---
        draw_dashboard(frame, avg_scores)

        # Emotion ko screen par display karte hain
        cv2.putText(frame, f"Emotion: {emotion}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        cv2.imshow('Emotion Detection Dashboard (MediaPipe)', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print("MediaPipe emotion detection stopped.")