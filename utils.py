import cv2
import numpy as np

LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
MOUTH_INDICES = [13, 14, 78, 308] 

def calculate_ear(landmarks, eye_indices):
    p2_p6 = np.linalg.norm(landmarks[eye_indices[1]] - landmarks[eye_indices[5]])
    p3_p5 = np.linalg.norm(landmarks[eye_indices[2]] - landmarks[eye_indices[4]])
    p1_p4 = np.linalg.norm(landmarks[eye_indices[0]] - landmarks[eye_indices[3]])
    
    ear = (p2_p6 + p3_p5) / (2.0 * p1_p4)
    return ear

def calculate_mar(landmarks, mouth_indices):
    p_top_p_bottom = np.linalg.norm(landmarks[mouth_indices[0]] - landmarks[mouth_indices[1]])
    p_left_p_right = np.linalg.norm(landmarks[mouth_indices[2]] - landmarks[mouth_indices[3]])
    mar = p_top_p_bottom / p_left_p_right
    return mar

def preprocess_frame(frame, output_size=(224, 224)):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    equalized_frame = cv2.equalizeHist(gray_frame)
    resized_frame = cv2.resize(equalized_frame, output_size, interpolation=cv2.INTER_AREA)
    return resized_frame
