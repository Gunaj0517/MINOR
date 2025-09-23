import cv2
import numpy as np
import mediapipe as mp

# ------------------- Gamma Correction -------------------
def adjust_gamma(image, gamma=1.5):
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255
                      for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(image, table)

def auto_gamma(image, target_brightness=120):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    if mean_brightness == 0:
        mean_brightness = 1
    gamma = target_brightness / mean_brightness
    gamma = max(0.5, min(2.5, gamma))  # clamp
    return adjust_gamma(image, gamma), gamma

# ------------------- Preprocessing -------------------
def preprocess_frame(frame):
    # Step 1: Gamma adjustment
    adjusted, gamma_used = auto_gamma(frame)

    # Step 2: CLAHE for local contrast enhancement
    lab = cv2.cvtColor(adjusted, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    enhanced = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    # Step 3: Light denoising
    denoised = cv2.GaussianBlur(enhanced, (3, 3), 0)

    # Step 4: Sharpen (to bring back details lost in blur)
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(denoised, -1, kernel)

    return sharpened, gamma_used

# ------------------- Mediapipe FaceMesh -------------------
mp_face = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils
face_mesh = mp_face.FaceMesh(
    max_num_faces=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# ------------------- Main Loop -------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (640, 480))

    processed, gamma_value = preprocess_frame(frame)

    # Convert to RGB for Mediapipe
    rgb_frame = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)

    # FaceMesh processing
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            mp_draw.draw_landmarks(
                processed, landmarks, mp_face.FACEMESH_CONTOURS,
                mp_draw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                mp_draw.DrawingSpec(color=(0, 0, 255), thickness=1))

    # Display gamma value
    cv2.putText(processed, f"Gamma: {gamma_value:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Driver Face Monitoring", processed)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()