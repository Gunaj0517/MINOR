import cv2
import mediapipe as mp
import numpy as np
import time
import pyautogui
import pygame

# --- Custom Modules Import ---
# utils.py se EAR/MAR logic import kar rahe hain
from utils import calculate_ear, calculate_mar, LEFT_EYE_INDICES, RIGHT_EYE_INDICES, MOUTH_INDICES
# sound_manager.py se sound logic import kar rahe hain
from sound_manager import init_sound, load_alarm, play_alarm, stop_alarm
# aapki gesture.py se classification function import kar rahe hain
try:
    from gesture import classify_gesture
except ImportError:
    print("Error: 'gesture.py' file not found.")
    print("Please make sure 'utils.py', 'sound_manager.py', and 'gesture.py' are in the same folder.")
    exit()

# ==============================================================================
# MASTER APP INITIALIZATION
# ==============================================================================

# --- Sound Setup ---
init_sound()
# Pehla alarm (Sleepy ke liye)
ALARM_SOUND_SLEEPY = load_alarm("alarm.mp3") 
# Doosra alarm (High Alert Drowsy ke liye)
ALARM_SOUND_DROWSY = load_alarm("wakeMe.mp3") 

# --- Mediapipe Models Setup ---
print("Initializing Mediapipe Models...")
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7)
    
print("Initialization completed.")

# --- Drowsiness Detection Parameters ---
EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 0.5
CONSECUTIVE_FRAMES_THRESHOLD = 60 # Frame rate tez tha, isliye isse badha diya hai
HEAD_TILT_THRESHOLD = -0.08 # Z-axis difference (forehead vs chin) to detect looking down

# --- Gesture Detection Parameters ---
last_gesture_time = 0
GESTURE_COOLDOWN = 1.0  # seconds

# --- Video Capture ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Unable to open the Webcam.")
    exit()

# --- Main App Variables ---
prev_time = 0
DROWSY_FRAMES_COUNTER = 0 
is_sleepy_alarm_playing = False # Alag alarm state
is_drowsy_alarm_playing = False # Alag alarm state
detected_gesture = "None" # Hold the last detected gesture

print("Webcam starting, Press 'q' to quit.")

# --- Gesture Action Function ---
# Yeh logic gesture.py ke main loop se liya gaya hai
def perform_gesture_action(gesture):
    """Executes a system command based on the detected gesture."""
    print(f"Performing action for: {gesture}")
    if gesture == "Thumbs Up":
        pyautogui.press("volumeup")
    elif gesture == "Thumbs Down":
        pyautogui.press("volumedown")
    elif gesture == "Open Palm":
        pyautogui.press("volumemute")
    elif gesture == "Victory":
        pyautogui.press("nexttrack")

# ==============================================================================
# MAIN APPLICATION LOOP
# ==============================================================================

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    
    # Drawing ke liye ek copy banate hain
    display_frame = frame.copy()
    h, w, _ = display_frame.shape

    # --- FPS Calculation ---
    current_time = time.time()
    fps = 0
    if prev_time > 0:
        fps = 1 / (current_time - prev_time)
    prev_time = current_time
    
    # --- Model Processing ---
    # Frame ko RGB mein convert karte hain (dono models ke liye)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame.flags.writeable = False # Performance optimization
    
    results_face = face_mesh.process(rgb_frame)
    results_hands = hands.process(rgb_frame)
    
    rgb_frame.flags.writeable = True # Frame ko wapas writeable banate hain

    # =================== DROWSINESS DETECTION ===================
    drowsy_status = "Awake"
    status_color = (0, 255, 0) # Green
    
    is_sleepy = False
    is_yawning = False
    is_looking_down = False
    
    if results_face.multi_face_landmarks:
        face_landmarks_list = results_face.multi_face_landmarks[0].landmark
        
        # Landmarks ko pixel coordinates mein convert karte hain
        pixel_landmarks = np.array([(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks_list])

        avg_ear = (calculate_ear(pixel_landmarks, LEFT_EYE_INDICES) + 
                   calculate_ear(pixel_landmarks, RIGHT_EYE_INDICES)) / 2.0
        mar = calculate_mar(pixel_landmarks, MOUTH_INDICES)
        
        # --- Head Tilt (Looking Down) Glitch Fix ---
        # Forehead (landmark 10) aur Chin (landmark 152) ke Z coordinate ko check kar rahe hain
        forehead_z = face_landmarks_list[10].z
        chin_z = face_landmarks_list[152].z
        z_diff = forehead_z - chin_z
        is_looking_down = z_diff < HEAD_TILT_THRESHOLD

        # Drowsiness ke landmarks (aankh/munh) draw karte hain
        mp_drawing.draw_landmarks(
            image=display_frame, 
            landmark_list=results_face.multi_face_landmarks[0], 
            connections=mp_face_mesh.FACEMESH_LEFT_EYE, 
            landmark_drawing_spec=None, 
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=1))
        mp_drawing.draw_landmarks(
            image=display_frame, 
            landmark_list=results_face.multi_face_landmarks[0], 
            connections=mp_face_mesh.FACEMESH_RIGHT_EYE, 
            landmark_drawing_spec=None, 
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=1))
        mp_drawing.draw_landmarks(
            image=display_frame, 
            landmark_list=results_face.multi_face_landmarks[0], 
            connections=mp_face_mesh.FACEMESH_LIPS, 
            landmark_drawing_spec=None, 
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1))

        # --- Drowsiness Logic (Time-based counter) ---
        # Agar aankhein band hain AUR user neeche nahi dekh raha, tabhi counter badhao
        if avg_ear < EAR_THRESHOLD and not is_looking_down:
            DROWSY_FRAMES_COUNTER += 1
        else:
            DROWSY_FRAMES_COUNTER = 0 # Reset agar aankh khuli ya user neeche dekh raha hai

        if DROWSY_FRAMES_COUNTER > CONSECUTIVE_FRAMES_THRESHOLD:
            is_sleepy = True # State 1: Sleepy

        # Yawn Detection
        yawn_status = "Yawn: No"
        if mar > MAR_THRESHOLD:
            is_yawning = True # State 2: Yawning
            yawn_status = "Yawn: Yes"
            
        # --- Alarm Logic (Sleepy vs Drowsy) ---
        if is_sleepy and is_yawning:
            # HIGH ALERT: DROWSY (Sleepy + Yawn)
            drowsy_status = "DROWSY!!"
            status_color = (255, 0, 255) # Magenta
            
            # Sleepy alarm band karke Drowsy alarm bajao
            if is_sleepy_alarm_playing:
                stop_alarm()
                is_sleepy_alarm_playing = False
            if ALARM_SOUND_DROWSY and not is_drowsy_alarm_playing:
                play_alarm(ALARM_SOUND_DROWSY)
                is_drowsy_alarm_playing = True

        elif is_sleepy:
            # MEDIUM ALERT: SLEEPY (Eyes closed)
            drowsy_status = "SLEEPY"
            status_color = (0, 0, 255) # Red
            
            # Sirf Sleepy alarm bajao (agar high alert pehle se nahi baj raha)
            if not is_drowsy_alarm_playing and ALARM_SOUND_SLEEPY and not is_sleepy_alarm_playing:
                play_alarm(ALARM_SOUND_SLEEPY)
                is_sleepy_alarm_playing = True
        
        else:
            # NO ALERT: AWAKE
            drowsy_status = "Awake"
            status_color = (0, 255, 0) # Green
            
            # Saare alarms band karo
            if is_sleepy_alarm_playing:
                stop_alarm()
                is_sleepy_alarm_playing = False
            if is_drowsy_alarm_playing:
                stop_alarm()
                is_drowsy_alarm_playing = False
            
        # Drowsiness Text display
        cv2.putText(display_frame, f"EAR: {avg_ear:.2f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(display_frame, f"MAR: {mar:.2f}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(display_frame, yawn_status, (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(display_frame, f"Frames Closed: {DROWSY_FRAMES_COUNTER}", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)
        # Glitch fix ka status
        tilt_status_text = f"Looking Down: {is_looking_down}"
        tilt_color = (0, 165, 255) if is_looking_down else (255, 255, 255)
        cv2.putText(display_frame, tilt_status_text, (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.9, tilt_color, 2)


    # =================== GESTURE DETECTION ===================
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            # Haath ke landmarks draw karte hain
            mp_drawing.draw_landmarks(
                display_frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS)
            
            # Gesture ko classify karte hain (gesture.py se)
            gesture = classify_gesture(hand_landmarks.landmark)

            if gesture:
                detected_gesture = gesture # Status mein dikhane ke liye store karte hain
                # Cooldown check
                if current_time - last_gesture_time > GESTURE_COOLDOWN:
                    perform_gesture_action(gesture) # Action perform karte hain
                    last_gesture_time = current_time

    # =================== FINAL DISPLAY ===================
    
    # FPS Display
    cv2.putText(display_frame, f'FPS: {int(fps)}', (w - 150, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Drowsiness Status Display
    cv2.putText(display_frame, f"STATUS: {drowsy_status}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3)
    
    # Gesture Status Display
    cv2.putText(display_frame, f"Gesture: {detected_gesture}", (w - 300, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Final combined window show karte hain
    cv2.imshow('Driver Monitoring System - MASTER', display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ==============================================================================
# CLEANUP
# ==============================================================================
stop_alarm()
cap.release()
face_mesh.close()
hands.close()
cv2.destroyAllWindows()
print("Webcam closed. Master application stopped.")

