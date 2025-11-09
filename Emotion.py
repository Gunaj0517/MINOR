import cv2
from fer import FER

detector = FER(mtcnn=True)
cap = cv2.VideoCapture(0)
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = detector.detect_emotions(frame)
    if result:
        (x, y, w, h) = result[0]["box"]
        emotions = result[0]["emotions"]
        top_emotion = max(emotions, key=emotions.get)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, f"{top_emotion} {emotions[top_emotion]:.2f}", 
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    cv2.imshow("FER Emotion Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
