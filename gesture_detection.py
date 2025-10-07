import cv2
import mediapipe as mp
import pyautogui
import collections
import time
import math

# ----------------- Setup -----------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
positions = collections.deque(maxlen=10)  # For swipe detection
last_action_time = 0
GESTURE_COOLDOWN = 2  # seconds

# ---------- Helper Functions ----------
def fingers_up(hand_landmarks):
    """Return a list of finger states: 1=extended, 0=folded"""
    tips = [mp_hands.HandLandmark.THUMB_TIP,
            mp_hands.HandLandmark.INDEX_FINGER_TIP,
            mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
            mp_hands.HandLandmark.RING_FINGER_TIP,
            mp_hands.HandLandmark.PINKY_TIP]
    
    pip = [mp_hands.HandLandmark.THUMB_IP,
           mp_hands.HandLandmark.INDEX_FINGER_PIP,
           mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
           mp_hands.HandLandmark.RING_FINGER_PIP,
           mp_hands.HandLandmark.PINKY_PIP]

    states = []
    for i in range(1, 5):  # index to pinky
        if hand_landmarks.landmark[tips[i]].y < hand_landmarks.landmark[pip[i]].y:
            states.append(1)
        else:
            states.append(0)
    # Thumb (x-axis)
    if hand_landmarks.landmark[tips[0]].x > hand_landmarks.landmark[pip[0]].x:
        states.insert(0, 1)
    else:
        states.insert(0, 0)
    return states  # [thumb, index, middle, ring, pinky]

def distance(a, b):
    """Euclidean distance between two landmarks"""
    return math.hypot(a.x - b.x, a.y - b.y)

def detect_static_gesture(hand_landmarks):
    """Detect common gestures based on finger states and positions"""
    states = fingers_up(hand_landmarks)
    
    # Pinch detection (thumb + index close)
    pinch_distance = distance(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP],
                              hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP])
    if pinch_distance < 0.04:
        return "Pinch"
    
    # Victory sign (index + middle up)
    if states == [0,1,1,0,0]:
        return "Victory"
    
    # Standard gestures
    if states == [1,1,1,1,1]:
        return "Open Palm"
    elif states == [0,0,0,0,0]:
        return "Fist"
    elif states == [1,0,0,0,0]:
        return "Thumbs Up"
    elif states == [0,0,0,0,1]:
        return "Thumbs Down"
    
    return None

def detect_swipe(positions, threshold=0.015):
    """Detect swipe left/right based on average wrist x movement"""
    if len(positions) < 5:
        return None
    deltas = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
    avg_delta = sum(deltas) / len(deltas)
    
    if avg_delta > threshold:
        positions.clear()
        return "Swipe Right"
    elif avg_delta < -threshold:
        positions.clear()
        return "Swipe Left"
    return None

def perform_action(gesture):
    """Map gestures to infotainment actions with cooldown"""
    global last_action_time
    current_time = time.time()
    if current_time - last_action_time < GESTURE_COOLDOWN:
        return
    last_action_time = current_time

    if gesture == "Open Palm":
        pyautogui.press('space')  # Play/Pause
        print("▶️ Play/Pause")
    elif gesture == "Fist":
        pyautogui.press('volumemute')  # Mute / Stop
        print("🔇 Mute/Stop")
    elif gesture == "Thumbs Up":
        pyautogui.hotkey('ctrl', 'up')  # Volume Up
        print("🔊 Volume Up")
    elif gesture == "Thumbs Down":
        pyautogui.hotkey('ctrl', 'down')  # Volume Down
        print("🔉 Volume Down")
    elif gesture == "Swipe Right":
        pyautogui.hotkey('ctrl', 'right')  # Next Track
        print("⏭ Next Track")
    elif gesture == "Swipe Left":
        pyautogui.hotkey('ctrl', 'left')  # Previous Track
        print("⏮ Previous Track")
    elif gesture == "Victory":
        pyautogui.hotkey('win', 'r')
        time.sleep(0.2)
        pyautogui.typewrite("https://www.google.com/maps\n")
        print("🗺️ Launch Navigation")
    elif gesture == "Pinch":
        pyautogui.hotkey('ctrl', 'shift', 'up')  # Fine volume up
        print("🔊 Volume Up (fine)")

# ---------- Main Loop ----------
with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        gesture = None
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Static gestures
                gesture = detect_static_gesture(hand_landmarks)

                # Swipe detection
                x = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x
                positions.append(x)
                swipe_gesture = detect_swipe(positions)
                if swipe_gesture:
                    gesture = swipe_gesture

        if gesture:
            perform_action(gesture)
            cv2.putText(frame, f"Gesture: {gesture}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        cv2.imshow("Gesture Infotainment Control", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
