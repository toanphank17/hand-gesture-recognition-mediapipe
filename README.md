# ✋ Hand Gesture Recognition Using MediaPipe

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python\&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Computer_Vision-orange)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Lite-FF6F00?logo=tensorflow\&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-5C3EE8?logo=opencv)
![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)

Real-time hand gesture recognition system built with **MediaPipe**, **TensorFlow Lite**, and **OpenCV**. This project supports both **static hand sign recognition** and **dynamic finger gesture recognition**, making it suitable for Human-Computer Interaction (HCI), sign language research, robotics, and embedded AI applications.

---

## 📸 Features

* Real-time hand tracking using MediaPipe
* Static hand sign classification
* Dynamic finger motion recognition
* Lightweight TensorFlow Lite inference
* Custom dataset collection tools
* Support for training new gestures
* Optimized for Raspberry Pi and low-power devices
* FPS monitoring for performance evaluation

---

## 🎥 Demo

<p align="center">
  <img src="docs/demo.gif" width="700">
</p>

Example gestures:

| Gesture           | Description             |
| ----------------- | ----------------------- |
| Open Palm         | Open hand               |
| Fist              | Closed hand             |
| Pointing          | Index finger pointing   |
| Clockwise         | Circular motion         |
| Counter Clockwise | Reverse circular motion |
| Moving            | Finger movement         |

---

## 🏗️ System Architecture

```text
Camera
   │
   ▼
MediaPipe Hand Tracking
   │
   ├── Hand Landmark Extraction
   │
   ├── Keypoint Preprocessing
   │      │
   │      ▼
   │   Static Gesture Classifier
   │
   └── Point History Buffer
          │
          ▼
     Dynamic Gesture Classifier

          ▼
      Gesture Output
```

---

## 📋 Requirements

### Software

* Python 3.11+
* OpenCV
* MediaPipe
* TensorFlow
* NumPy

### Raspberry Pi Dependencies

```bash
sudo apt update

sudo apt install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libportaudio2
```

---

## 🚀 Installation

### Clone Repository

```bash
git clone https://github.com/<your-username>/hand-gesture-recognition.git

cd hand-gesture-recognition
```

### Create Virtual Environment

```bash
python -m venv venv

source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

Start webcam inference:

```bash
python app.py
```

### Optional Parameters

| Argument                   | Description          | Default |
| -------------------------- | -------------------- | ------- |
| --device                   | Camera device index  | 0       |
| --width                    | Capture width        | 960     |
| --height                   | Capture height       | 540     |
| --min_detection_confidence | Detection confidence | 0.5     |
| --min_tracking_confidence  | Tracking confidence  | 0.5     |

Example:

```bash
python app.py \
    --device 0 \
    --width 1280 \
    --height 720
```

---

## 📂 Project Structure

```text
.
├── app.py
├── keypoint_classification.ipynb
├── point_history_classification.ipynb
│
├── model
│   ├── keypoint_classifier
│   │   ├── keypoint.csv
│   │   ├── keypoint_classifier.hdf5
│   │   ├── keypoint_classifier.tflite
│   │   └── keypoint_classifier_label.csv
│   │
│   └── point_history_classifier
│       ├── point_history.csv
│       ├── point_history_classifier.hdf5
│       ├── point_history_classifier.tflite
│       └── point_history_classifier_label.csv
│
└── utils
    └── cvfpscalc.py
```

---

## 🏋️ Training Custom Hand Sign Models

### Step 1: Collect Training Data

Press:

```text
k
```

to enter Key Point Logging Mode.

Then press:

```text
0 - 9
```

to save landmark data.

Generated data will be stored in:

```text
model/keypoint_classifier/keypoint.csv
```

Default classes:

| ID | Label       |
| -- | ----------- |
| 0  | Open Hand   |
| 1  | Closed Hand |
| 2  | Pointing    |

---

### Step 2: Train Model

Open:

```text
keypoint_classification.ipynb
```

and execute all cells.

To add new classes:

1. Update `NUM_CLASSES`
2. Modify

```text
model/keypoint_classifier/keypoint_classifier_label.csv
```

---

## 👆 Training Dynamic Finger Gesture Models

### Step 1: Collect Motion Data

Press:

```text
h
```

to enter Point History Logging Mode.

Press:

```text
0 - 9
```

to record movement sequences.

Data will be stored in:

```text
model/point_history_classifier/point_history.csv
```

Default classes:

| ID | Label             |
| -- | ----------------- |
| 0  | Stationary        |
| 1  | Clockwise         |
| 2  | Counter Clockwise |
| 3  | Moving            |

---

### Step 2: Train Model

Open:

```text
point_history_classification.ipynb
```

and run all cells.

For LSTM-based training:

```python
use_lstm = True
```

---

## 📈 Performance

| Device                  | FPS   |
| ----------------------- | ----- |
| Desktop PC (RTX Series) | 60+   |
| Laptop CPU              | 25-45 |
| Raspberry Pi 5          | 20-30 |

Actual performance may vary depending on camera resolution and model complexity.

---

## 🔬 Applications

* Sign Language Recognition
* Human Computer Interaction (HCI)
* Smart Home Control
* Robotics Interfaces
* Educational Projects
* Gesture-Based Gaming
* Embedded AI Systems

---

## 🙏 Acknowledgements

This project is based on the excellent work of:

* Kazuhito Takahashi
* Nikita Kiselov
* MediaPipe Team
* TensorFlow Lite Team

Original project:

https://github.com/kinivi/hand-gesture-recognition-mediapipe

---

## 📄 License

This project is licensed under the Apache License 2.0.

See the LICENSE file for details.
