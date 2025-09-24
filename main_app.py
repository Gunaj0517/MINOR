import cv2
import mediapipe as mp
import numpy as np
# utils.py se saare functions import kar rahe hain
from utils import calculate_ear, calculate_mar, preprocess_frame, LEFT_EYE_INDICES, RIGHT_EYE_INDICES, MOUTH_INDICES

# --- MediaPipe Initialization ---
print("MediaPipe Face Mesh ko initialize kiya ja raha hai...")
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

print("Initialization poora hua.")

# --- Drowsiness Detection Parameters ---
EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 0.5

# --- Main Video Loop ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Webcam nahi khul saka.")
    exit()

print("Webcam feed shuru ho raha hai. 'q' dabakar band karein.")

# Bada tile size set kar rahe hain (3 columns, 2 rows)
TILE_SIZE = (480, 360)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    
    # --- 1. Landmark Detection ---
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    # --- 2. Saari 6 Tiles Taiyar Karna ---
    
    # Tile 1: Plain Original Frame
    tile1_original = cv2.resize(frame, TILE_SIZE)
    cv2.putText(tile1_original, '1. Original', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Tile 2: Standard Processed Frame
    processed_frame = preprocess_frame(frame, output_size=TILE_SIZE)
    tile2_processed = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)
    cv2.putText(tile2_processed, '2. Processed', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Tile 3: More Blurred Frame
    blurred_frame = cv2.GaussianBlur(frame, (21, 21), 0)
    tile3_blurred = cv2.resize(blurred_frame, TILE_SIZE)
    cv2.putText(tile3_blurred, '3. Blurred', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Tile 4: Sharpened (De-pixeled) Frame
    sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened_frame = cv2.filter2D(frame, -1, sharpen_kernel)
    tile4_sharpened = cv2.resize(sharpened_frame, TILE_SIZE)
    cv2.putText(tile4_sharpened, '4. Sharpened', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Tile 5: Full Mesh on Original
    tile5_full_mesh = cv2.resize(frame.copy(), TILE_SIZE)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=tile5_full_mesh,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
    cv2.putText(tile5_full_mesh, '5. Full Mesh', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Tile 6: MAIN TILE - Drowsiness Detection
    tile6_main = cv2.resize(frame.copy(), TILE_SIZE)
    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        
        h, w, _ = tile6_main.shape
        pixel_landmarks = np.array([(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark])

        avg_ear = (calculate_ear(pixel_landmarks, LEFT_EYE_INDICES) + calculate_ear(pixel_landmarks, RIGHT_EYE_INDICES)) / 2.0
        mar = calculate_mar(pixel_landmarks, MOUTH_INDICES)
        
        drowsy_status = "Awake"
        status_color = (0, 255, 0)
        if avg_ear < EAR_THRESHOLD:
            drowsy_status = "Drowsy"
            status_color = (0, 0, 255)
        
        yawn_status = "Yawn: No"
        if mar > MAR_THRESHOLD:
            yawn_status = "Yawn: Yes"

        cv2.putText(tile6_main, f"EAR: {avg_ear:.2f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(tile6_main, f"MAR: {mar:.2f}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(tile6_main, drowsy_status, (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.5, status_color, 3)
        cv2.putText(tile6_main, yawn_status, (20, 230), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.putText(tile6_main, '6. Drowsiness Detection', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # --- 3. Grid ko Combine karke Dikhana ---
    top_row = np.hstack([tile1_original, tile2_processed, tile3_blurred])
    bottom_row = np.hstack([tile4_sharpened, tile5_full_mesh, tile6_main])
    combined_view = np.vstack([top_row, bottom_row])
    
    cv2.imshow('Driver Monitoring System - Dashboard', combined_view)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
face_mesh.close()
cv2.destroyAllWindows()
print("Webcam feed band ho gaya.")

