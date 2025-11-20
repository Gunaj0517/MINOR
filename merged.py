import cv2
import mediapipe as mp
import numpy as np
import time
import pyautogui
import pygame
import csv
from fer import FER
from utils import calculate_ear, calculate_mar, LEFT_EYE_INDICES, RIGHT_EYE_INDICES, MOUTH_INDICES
from sound_manager import init_sound, load_alarm, play_alarm, stop_alarm

try:
    from gesture import classify_gesture
except ImportError:
    print("Error: gesture.py missing!")
    exit()

# INITIALIZATION
init_sound()
ALARM_SOUND_SLEEPY = load_alarm("alarm.mp3")
ALARM_SOUND_DROWSY = load_alarm("wakeMe.mp3")

mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

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

emotion_detector = FER(mtcnn=True)

# PARAMETERS 
EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 0.5
CONSECUTIVE_FRAMES_THRESHOLD = 15
HEAD_TILT_THRESHOLD = -0.08

last_gesture_time = 0
GESTURE_COOLDOWN = 1.0

DROWSY_FRAMES_COUNTER = 0
is_sleepy_alarm_playing = False
is_drowsy_alarm_playing = False
detected_gesture = "None"

true_labels = []   
pred_labels = []   

gesture_mapping = {
    "Thumbs Up": 1,
    "Thumbs Down": 2,
    "Open Palm": 3,
    "Victory": 4
}
label_to_name = {v: k for k, v in gesture_mapping.items()}

try:
    with open("gesture_accuracy.csv", "r") as f:
        reader = csv.reader(f)
        header = next(reader, None)  # Skip header

        for row in reader:
            if len(row) >= 2:
                true_labels.append(int(row[0]))
                pred_labels.append(int(row[1]))

    print(f"[LOADED] {len(true_labels)} previous gesture samples loaded from gesture_accuracy.csv")

except FileNotFoundError:
    print("No previous gesture_accuracy.csv found. Starting fresh.")

def compute_accuracy():
    if not true_labels:
        return "No accuracy data recorded yet."

    total = len(true_labels)
    correct = sum(1 for t, p in zip(true_labels, pred_labels) if t == p)
    acc = (correct / total) * 100
    matrix = np.zeros((4, 4), dtype=int)

    for t, p in zip(true_labels, pred_labels):
        if 1 <= t <= 4 and 1 <= p <= 4:
            matrix[t-1][p-1] += 1

    text = f"\n=== ACCURACY REPORT ===\n" \
           f"Total Samples: {total}\n" \
           f"Correct: {correct}\n" \
           f"Accuracy: {acc:.2f}%\n\n" \
           f"Confusion Matrix (rows=true 1..4, cols=pred 1..4):\n{matrix}\n"

    per_class = ""
    for cls in range(1, 5):
        idxs = [i for i, t in enumerate(true_labels) if t == cls]
        if idxs:
            cls_correct = sum(1 for i in idxs if pred_labels[i] == cls)
            per_class += f"{label_to_name[cls]}: {cls_correct}/{len(idxs)} correct\n"
        else:
            per_class += f"{label_to_name[cls]}: No samples\n"

    return text + "\n" + per_class


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Webcam could not open.")
    exit()

prev_time = 0
print("Running... press 'q' to quit. Keys: 1(ThumbsUp) 2(ThumbsDown) 3(OpenPalm) 4(Victory) A(accuracy) S(save CSV)")

# GESTURE ACTION FUNCTION
def perform_gesture_action(gesture):
    print("Action for:", gesture)
    if gesture == "Thumbs Up":
        pyautogui.press("volumeup")
    elif gesture == "Thumbs Down":
        pyautogui.press("volumedown")
    elif gesture == "Open Palm":
        pyautogui.press("volumemute")
    elif gesture == "Victory":
        pyautogui.press("nexttrack")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    display_frame = frame.copy()
    h, w, _ = frame.shape
    detected_gesture = "None"

    # FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time else 0
    prev_time = current_time

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame.flags.writeable = False

    results_face = face_mesh.process(rgb_frame)
    results_hands = hands.process(rgb_frame)

    rgb_frame.flags.writeable = True

    # EMOTION DETECTION
    emotion_text = "Emotion: None"
    emotion_color = (255, 255, 255)

    emotion_results = emotion_detector.detect_emotions(frame)
    if emotion_results:
        emotions = emotion_results[0]["emotions"]
        top_emotion = max(emotions, key=emotions.get)
        emotion_text = f"Emotion: {top_emotion} ({emotions[top_emotion]:.2f})"
        emotion_color = (0, 255, 255)

    # DROWSINESS DETECTION
    drowsy_status = "Awake"
    status_color = (0, 255, 0)
    is_sleepy = False
    is_yawning = False
    is_looking_down = False

    if results_face.multi_face_landmarks:
        face_landmarks = results_face.multi_face_landmarks[0].landmark
        pixel_landmarks = np.array([(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks])

        avg_ear = (calculate_ear(pixel_landmarks, LEFT_EYE_INDICES) +
                   calculate_ear(pixel_landmarks, RIGHT_EYE_INDICES)) / 2

        mar = calculate_mar(pixel_landmarks, MOUTH_INDICES)

        forehead_z = face_landmarks[10].z
        chin_z = face_landmarks[152].z
        z_diff = forehead_z - chin_z
        is_looking_down = z_diff < HEAD_TILT_THRESHOLD

        if avg_ear < EAR_THRESHOLD and not is_looking_down:
            DROWSY_FRAMES_COUNTER += 1
        else:
            DROWSY_FRAMES_COUNTER = 0

        if DROWSY_FRAMES_COUNTER > CONSECUTIVE_FRAMES_THRESHOLD:
            is_sleepy = True

        if mar > MAR_THRESHOLD:
            is_yawning = True

        # SLEEPY + YAWNING = HIGH ALERT
        if is_sleepy and is_yawning:
            drowsy_status = "DROWSY!!"
            status_color = (255, 0, 255)

            if is_sleepy_alarm_playing:
                stop_alarm()
                is_sleepy_alarm_playing = False

            if not is_drowsy_alarm_playing:
                play_alarm(ALARM_SOUND_DROWSY)
                is_drowsy_alarm_playing = True

        elif is_sleepy:
            drowsy_status = "SLEEPY"
            status_color = (0, 0, 255)

            if not is_sleepy_alarm_playing:
                play_alarm(ALARM_SOUND_SLEEPY)
                is_sleepy_alarm_playing = True

        else:
            drowsy_status = "Awake"
            status_color = (0, 255, 0)

            stop_alarm()
            is_sleepy_alarm_playing = False
            is_drowsy_alarm_playing = False

        cv2.putText(display_frame, f"EAR: {avg_ear:.2f}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        cv2.putText(display_frame, f"MAR: {mar:.2f}", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        cv2.putText(display_frame, f"Yawn: {is_yawning}", (20, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        cv2.putText(display_frame, f"Frames Closed: {DROWSY_FRAMES_COUNTER}", (20, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)

        cv2.putText(display_frame, f"Looking Down: {is_looking_down}", (20, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)

        cv2.putText(display_frame, emotion_text, (20, 280),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, emotion_color, 2)

    # HAND GESTURE DETECTION
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            mp_drawing.draw_landmarks(display_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            gesture = classify_gesture(hand_landmarks.landmark)
            if gesture:
                detected_gesture = gesture

                if current_time - last_gesture_time > GESTURE_COOLDOWN:
                    perform_gesture_action(gesture)
                    last_gesture_time = current_time

    cv2.putText(display_frame, f"FPS: {int(fps)}", (w - 150, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(display_frame, f"STATUS: {drowsy_status}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3)

    cv2.putText(display_frame, f"Gesture: {detected_gesture}", (w - 320, h - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.putText(display_frame, "1 : ThumbsUp 2 : ThumbsDown 3 : OpenPalm 4 : Victory  A : Acc  S : Save", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow("Combined Driver Monitoring System", display_frame)
    key = cv2.waitKey(1) & 0xFF

    if key in [ord('1'), ord('2'), ord('3'), ord('4')]:
        true = int(chr(key))
        true_labels.append(true)
        pred = gesture_mapping.get(detected_gesture, 0)  
        pred_labels.append(pred)
        print(f"[RECORDED] True={true} ({label_to_name[true]}), Pred={pred} ({detected_gesture})")

    elif key == ord('a') or key == ord('A'):
        print(compute_accuracy())

    elif key == ord('s') or key == ord('S'):
        file_exists = False
        try:
            with open("gesture_accuracy.csv", "r") as f:
                file_exists = True
        except FileNotFoundError:
            pass

        with open("gesture_accuracy.csv", "a", newline="") as f:
            writer = csv.writer(f)

            if not file_exists:
                writer.writerow(["True", "Predicted"])

            for t, p in zip(true_labels, pred_labels):
                writer.writerow([t, p])
        print("Appended new samples to: gesture_accuracy.csv")

    elif key == ord('q') or key == ord('Q'):
        break

stop_alarm()
cap.release()
face_mesh.close()
hands.close()
cv2.destroyAllWindows()