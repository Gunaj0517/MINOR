# Driver Monitoring System 🚘👀

An AI-powered driver safety system that integrates **Drowsiness Detection**, **Emotion Recognition**, and **Gesture Control** into a single, real-time pipeline. The project uses **OpenCV**, **MediaPipe**, and **Deep Learning models** to monitor driver state and provide timely alerts to prevent accidents.

---

## 📌 Features

* **Drowsiness Detection** – Uses Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR) to detect prolonged eye closure, yawning, and head nodding.
* **Emotion Detection** – Classifies driver emotions (happy, sad, angry, stressed, neutral) using a CNN trained on FER-2013 dataset.
* **Gesture Control** – Enables non-intrusive infotainment control (volume, calls, navigation) using MediaPipe Hand gestures.
* **Real-time Alerts** – Generates audio/visual warnings when unsafe driver states are detected.
* **Cost-Effective** – Works with just a webcam, no additional sensors required.

---

## 🛠️ Tech Stack

* **Language:** Python 3
* **Libraries/Frameworks:** OpenCV, MediaPipe, TensorFlow/Keras, NumPy
* **Dataset:** FER-2013 (for emotion detection)
* **IDE:** Jupyter Notebook / VS Code

---

## 📂 Project Structure

```
Driver-Monitoring-System/
│── data/                # Datasets (FER-2013, sample videos/images)
│── models/              # Pre-trained/fine-tuned CNN models
│── src/
│   ├── drowsiness.py    # EAR & MAR based detection
│   ├── emotion.py       # Emotion classification using CNN
│   ├── gesture.py       # Hand gesture recognition
│   ├── main.py          # Integrated pipeline
│── requirements.txt     # Dependencies
│── README.md            # Project documentation
```

---

## 🚀 How to Run

1. Clone the repository

   ```bash
   git clone https://github.com/your-username/driver-monitoring-system.git
   cd driver-monitoring-system
   ```
2. Install dependencies

   ```bash
   pip install -r requirements.txt
   ```
3. Run the system

   ```bash
   python src/main.py
   ```

---

## 📊 Experimental Results

* Drowsiness Detection Accuracy: **91%**
* Emotion Detection Accuracy: **82%**
* Gesture Recognition Accuracy: **95%**
* Average Frame Rate: **15–20 FPS (CPU)**

---

## 🔮 Future Improvements

* Adaptive preprocessing for extreme lighting conditions
* Dataset expansion for diverse emotions & gestures
* GPU acceleration for higher FPS
* Integration with IoT/automotive hardware for deployment

---

## 👥 Team

Developed by **\[Your Team Name]** – 3rd Year CSE Students

* Member 1
* Member 2
* Member 3
* Member 4

---

## 📜 License

This project is licensed under the MIT License – feel free to use and modify.

---
