import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time

# ----------------- Initialize MediaPipe -----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# ----------------- Helper Functions -----------------
def calc_angle(a, b, c):
    a = np.array(a); b = np.array(b); c = np.array(c)
    ba = a - b; bc = c - b
    cos_angle = np.dot(ba, bc)/(np.linalg.norm(ba)*np.linalg.norm(bc)+1e-6)
    angle = np.arccos(np.clip(cos_angle,-1.0,1.0))
    return np.degrees(angle)

def finger_states(landmarks):
    states = {}
    finger_tips = [8,12,16,20]
    finger_pips = [6,10,14,18]
    finger_mcp = [5,9,13,17]
    names = ['index','middle','ring','pinky']
    for tip,pip,mcp,name in zip(finger_tips,finger_pips,finger_mcp,names):
        angle = calc_angle([landmarks[mcp].x,landmarks[mcp].y],
                           [landmarks[pip].x,landmarks[pip].y],
                           [landmarks[tip].x,landmarks[tip].y])
        states[name] = 1 if angle > 160 else 0

    # Thumb vertical (for up/down)
    wrist_y = landmarks[0].y
    tip_y = landmarks[4].y
    mcp_y = landmarks[2].y
    if tip_y < wrist_y and tip_y < mcp_y:
        states['thumb_vert'] = 1
    elif tip_y > wrist_y and tip_y > mcp_y:
        states['thumb_vert'] = -1
    else:
        states['thumb_vert'] = 0
    return states

def classify_gesture(landmarks):
    s = finger_states(landmarks)
    t_vert = s['thumb_vert']
    i = s['index']; m = s['middle']; r = s['ring']; p = s['pinky']

    # Thumb Up / Down
    if t_vert == 1 and i == 0 and m == 0 and r == 0 and p == 0:
        return "Thumbs Up"
    elif t_vert == -1 and i == 0 and m == 0 and r == 0 and p == 0:
        return "Thumbs Down"

    # Open Palm
    if i == 1 and m == 1 and r == 1 and p == 1:
        return "Open Palm"

    # Victory
    if i == 1 and m == 1 and r == 0 and p == 0:
        return "Victory"

    return None

# ----------------- Gesture Debounce -----------------
last_action_time = 0
action_delay = 1.0  # seconds

# ----------------- Camera -----------------
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    gesture = None

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
            gesture = classify_gesture(hand.landmark)
            if gesture: break

    current_time = time.time()
    if gesture and current_time - last_action_time > action_delay:
        # ----------------- Actual Controls -----------------
        if gesture == "Thumbs Up":
            pyautogui.press("volumeup")
        elif gesture == "Thumbs Down":
            pyautogui.press("volumedown")
        elif gesture == "Open Palm":
            pyautogui.press("volumemute")
        elif gesture == "Victory":
            pyautogui.press("nexttrack")
        last_action_time = current_time

        print("Detected Gesture:", gesture)

    if gesture:
        cv2.putText(frame, f"Gesture: {gesture}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Car Gesture Control", frame)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()
