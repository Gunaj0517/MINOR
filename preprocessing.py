import cv2
import numpy as np
import mediapipe as mp

# ---------------- Preprocessing ----------------
def preprocess_frame(frame):
    # Resize + mild blur for stability
    frame = cv2.resize(frame, (640, 480))
    frame = cv2.GaussianBlur(frame, (3, 3), 0)
    return frame

# ---------------- Mediapipe FaceMesh ----------------
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    max_num_faces=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# Eye + Mouth landmark indices
LEFT_EYE = [33, 133, 160, 159, 158, 153, 144, 145]
RIGHT_EYE = [362, 263, 387, 386, 385, 380, 373, 374]
MOUTH = [61, 291, 308, 78, 95, 324, 14, 17, 13]

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    processed = preprocess_frame(frame)
    rgb_frame = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = processed.shape

            # Draw left eye
            for idx in LEFT_EYE:
                x, y = int(face_landmarks.landmark[idx].x * w), int(face_landmarks.landmark[idx].y * h)
                cv2.circle(processed, (x, y), 2, (0, 255, 0), -1)

            # Draw right eye
            for idx in RIGHT_EYE:
                x, y = int(face_landmarks.landmark[idx].x * w), int(face_landmarks.landmark[idx].y * h)
                cv2.circle(processed, (x, y), 2, (0, 255, 0), -1)

            # Draw mouth
            for idx in MOUTH:
                x, y = int(face_landmarks.landmark[idx].x * w), int(face_landmarks.landmark[idx].y * h)
                cv2.circle(processed, (x, y), 2, (0, 0, 255), -1)

    cv2.imshow("Eyes + Mouth Landmarks", processed)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
