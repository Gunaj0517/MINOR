import cv2
import mediapipe as mp
import numpy as np
import time
import pyautogui
import pygame
from fer import FER

# --- Custom Modules ---
from utils import calculate_ear, calculate_mar, LEFT_EYE_INDICES, RIGHT_EYE_INDICES, MOUTH_INDICES
from sound_manager import init_sound, load_alarm, play_alarm, stop_alarm
try:
    from gesture import classify_gesture
except ImportError:
    print("Error: 'gesture.py' file not found.")
    exit()

# ========================================================================
# INITIALIZATION
# ========================================================================

# Emotion detector setup
print("Initializing FER Emotion Detector...")
emotion_detector = FER()  # uses Haar cascade in updated fer.py

# Sound system
init_sound()
ALARM_SOUND_SLEEPY = load_alarm("alarm.mp3")
ALARM_SOUND_DROWSY = load_alarm("wakeMe.mp3")

# Mediapipe setup
print("Initializing Mediapipe Models...")
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
print("Initialization completed.")

# ========================================================================
# CONSTANTS
# ========================================================================
EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 0.5
CONSECUTIVE_FRAMES_THRESHOLD = 25  # lower while testing e.g. 20
HEAD_TILT_THRESHOLD = -0.08
GESTURE_COOLDOWN = 1.0  # seconds

# ========================================================================
# MAIN LOOP VARIABLES
# ========================================================================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Unable to open the Webcam.")
    exit()

prev_time = 0
DROWSY_FRAMES_COUNTER = 0
is_sleepy_alarm_playing = False
is_drowsy_alarm_playing = False
detected_gesture = "None"
last_gesture_time = 0

print("Webcam starting... Press 'q' to quit.")

# ========================================================================
# GESTURE ACTION FUNCTION
# ========================================================================
def perform_gesture_action(gesture):
    print(f"Performing action for: {gesture}")
    if gesture == "Thumbs Up":
        pyautogui.press("volumeup")
    elif gesture == "Thumbs Down":
        pyautogui.press("volumedown")
    elif gesture == "Open Palm":
        pyautogui.press("volumemute")
    elif gesture == "Victory":
        pyautogui.press("nexttrack")

# Helper: connection drawing spec (no dots)
CONN_SPEC_EYE = mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=1)
CONN_SPEC_LIPS = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)

# ========================================================================
# MAIN APPLICATION LOOP
# ========================================================================
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    display_frame = frame.copy()
    h, w, _ = display_frame.shape

    # FPS Calculation
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
    prev_time = current_time

    # Convert to RGB for Mediapipe and FER
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame.flags.writeable = False
    results_face = face_mesh.process(rgb_frame)
    results_hands = hands.process(rgb_frame)
    rgb_frame.flags.writeable = True

    # ====================================================================
    # EMOTION DETECTION (FER)
    # ====================================================================
    emotion_text = "N/A"
    emotion_conf = 0.0
    emotion_box = None

    try:
        emotions = emotion_detector.detect_emotions(frame)  # FER expects BGR
    except Exception as e:
        emotions = []
        # optionally: print("FER error:", e)

    if emotions:
        (ex, ey, ew, eh) = emotions[0]["box"]
        emotion_scores = emotions[0]["emotions"]
        top_emotion = max(emotion_scores, key=emotion_scores.get)
        emotion_text = f"{top_emotion.capitalize()}"
        emotion_conf = emotion_scores[top_emotion]
        emotion_box = (ex, ey, ew, eh)
        # Draw emotion box
        cv2.rectangle(display_frame, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        cv2.putText(display_frame, f"{emotion_text} {emotion_conf:.2f}",
                    (ex, ey - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # ====================================================================
    # DROWSINESS DETECTION
    # ====================================================================
    drowsy_status = "Awake"
    status_color = (0, 255, 0)
    is_sleepy = False
    is_yawning = False
    is_looking_down = False

    if results_face and results_face.multi_face_landmarks:
        face_landmarks = results_face.multi_face_landmarks[0].landmark
        # pixel coords for EAR/MAR utils
        pixel_landmarks = np.array([(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks])

        # compute EAR and MAR
        try:
            left_ear = calculate_ear(pixel_landmarks, LEFT_EYE_INDICES)
            right_ear = calculate_ear(pixel_landmarks, RIGHT_EYE_INDICES)
            avg_ear = (left_ear + right_ear) / 2.0
        except Exception:
            avg_ear = 1.0  # default open eyes if calculation fails

        try:
            mar = calculate_mar(pixel_landmarks, MOUTH_INDICES)
        except Exception:
            mar = 0.0

        # head tilt z-diff guard
        try:
            forehead_z = face_landmarks[10].z
            chin_z = face_landmarks[152].z
            z_diff = forehead_z - chin_z
            is_looking_down = z_diff < HEAD_TILT_THRESHOLD
        except Exception:
            is_looking_down = False

        # Draw only connections (no landmark dots)
        mp_drawing.draw_landmarks(
            display_frame,
            results_face.multi_face_landmarks[0],
            mp_face_mesh.FACEMESH_LEFT_EYE,
            landmark_drawing_spec=None,
            connection_drawing_spec=CONN_SPEC_EYE,
        )
        mp_drawing.draw_landmarks(
            display_frame,
            results_face.multi_face_landmarks[0],
            mp_face_mesh.FACEMESH_RIGHT_EYE,
            landmark_drawing_spec=None,
            connection_drawing_spec=CONN_SPEC_EYE,
        )
        mp_drawing.draw_landmarks(
            display_frame,
            results_face.multi_face_landmarks[0],
            mp_face_mesh.FACEMESH_LIPS,
            landmark_drawing_spec=None,
            connection_drawing_spec=CONN_SPEC_LIPS,
        )

        # Drowsy frame counter logic
        if avg_ear < EAR_THRESHOLD and not is_looking_down:
            DROWSY_FRAMES_COUNTER += 1
        else:
            DROWSY_FRAMES_COUNTER = 0

        if DROWSY_FRAMES_COUNTER > CONSECUTIVE_FRAMES_THRESHOLD:
            is_sleepy = True

        if mar > MAR_THRESHOLD:
            is_yawning = True

        # Alarm logic
        if is_sleepy and is_yawning:
            drowsy_status = "DROWSY!!"
            status_color = (255, 0, 255)
            if is_sleepy_alarm_playing:
                stop_alarm()
                is_sleepy_alarm_playing = False
            if ALARM_SOUND_DROWSY and not is_drowsy_alarm_playing:
                play_alarm(ALARM_SOUND_DROWSY)
                is_drowsy_alarm_playing = True

        elif is_sleepy:
            drowsy_status = "SLEEPY"
            status_color = (0, 0, 255)
            if not is_drowsy_alarm_playing and ALARM_SOUND_SLEEPY and not is_sleepy_alarm_playing:
                play_alarm(ALARM_SOUND_SLEEPY)
                is_sleepy_alarm_playing = True
        else:
            drowsy_status = "Awake"
            status_color = (0, 255, 0)
            if is_sleepy_alarm_playing or is_drowsy_alarm_playing:
                stop_alarm()
                is_sleepy_alarm_playing = False
                is_drowsy_alarm_playing = False

        # Display Drowsiness Info
        cv2.putText(display_frame, f"EAR: {avg_ear:.2f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(display_frame, f"MAR: {mar:.2f}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(display_frame, f"Frames Closed: {DROWSY_FRAMES_COUNTER}", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,255), 2)
        cv2.putText(display_frame, f"Looking Down: {is_looking_down}", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,165,255), 2)
        cv2.putText(display_frame, f"Yawn: {'Yes' if is_yawning else 'No'}", (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

    else:
        # No face detected: reset frame counter so stale counts don't trigger
        DROWSY_FRAMES_COUNTER = 0
        avg_ear = 1.0
        mar = 0.0
        is_looking_down = False

    # ====================================================================
    # GESTURE DETECTION
    # ====================================================================
    if results_hands and results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            mp_drawing.draw_landmarks(display_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                      landmark_drawing_spec=None,
                                      connection_drawing_spec=mp_drawing.DrawingSpec(thickness=2))
            gesture = classify_gesture(hand_landmarks.landmark)
            if gesture:
                detected_gesture = gesture
                if current_time - last_gesture_time > GESTURE_COOLDOWN:
                    perform_gesture_action(gesture)
                    last_gesture_time = current_time

    # ====================================================================
    # LEFT SIDE TEXT (INCLUDING EMOTION)
    # ====================================================================
    cv2.putText(display_frame, f"STATUS: {drowsy_status}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 3)
    cv2.putText(display_frame, f"Emotion: {emotion_text} ({emotion_conf:.2f})", (20, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Bottom Right Info
    cv2.putText(display_frame, f"Gesture: {detected_gesture}", (w - 320, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    cv2.putText(display_frame, f'FPS: {int(fps)}', (w - 150, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Driver Monitoring System - Master", display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ====================================================================
# CLEANUP
# ====================================================================
stop_alarm()
cap.release()
face_mesh.close()
hands.close()
cv2.destroyAllWindows()
print("Application stopped.")
