import cv2
import mediapipe as mp
import numpy as np

from utils import calculate_ear, calculate_mar, preprocess_frame, LEFT_EYE_INDICES, RIGHT_EYE_INDICES, MOUTH_INDICES

print("Initializing MediaPipe...")
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

print("Initialization Finished.")

EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 0.5

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Unable to open the Webcam.")
    exit()

print("Starting Webcam, Press q to quit.")

TILE_SIZE = (640, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    tile1_original = cv2.resize(frame, TILE_SIZE)
    cv2.putText(tile1_original, '1. Original Feed', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    processed_frame = preprocess_frame(frame, output_size=TILE_SIZE)
    tile2_processed = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)
    cv2.putText(tile2_processed, '2. Processed Feed', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    tile3_full_mesh = cv2.resize(frame.copy(), TILE_SIZE)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=tile3_full_mesh, landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
    cv2.putText(tile3_full_mesh, '3. Full Face Mesh', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    tile4_main = cv2.resize(frame.copy(), TILE_SIZE)
    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        
        h, w, _ = tile4_main.shape
        pixel_landmarks = np.array([(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark])

        avg_ear = (calculate_ear(pixel_landmarks, LEFT_EYE_INDICES) + calculate_ear(pixel_landmarks, RIGHT_EYE_INDICES)) / 2.0
        mar = calculate_mar(pixel_landmarks, MOUTH_INDICES)

        drowsy_status = "Awake"
        status_color = (0, 255, 0) # Green
        if avg_ear < EAR_THRESHOLD:
            drowsy_status = "Drowsy"
            status_color = (0, 0, 255) # Red
        
        yawn_status = "Yawn: No"
        if mar > MAR_THRESHOLD:
            yawn_status = "Yawn: Yes"

        cv2.putText(tile4_main, f"EAR: {avg_ear:.2f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        cv2.putText(tile4_main, f"MAR: {mar:.2f}", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        cv2.putText(tile4_main, drowsy_status, (20, 220), cv2.FONT_HERSHEY_SIMPLEX, 2, status_color, 4)
        cv2.putText(tile4_main, yawn_status, (20, 280), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)

    cv2.putText(tile4_main, '4. Drowsiness Detection', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    top_row = np.hstack([tile1_original, tile2_processed])
    bottom_row = np.hstack([tile3_full_mesh, tile4_main])
    combined_view = np.vstack([top_row, bottom_row])
    
    cv2.imshow('Driver Monitoring System - Dashboard', combined_view)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
face_mesh.close()
cv2.destroyAllWindows()
print("Webcam closed.")
