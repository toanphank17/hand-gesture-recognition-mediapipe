Markdown# Hand Gesture Recognition Using MediaPipe

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Raspberry Pi](https://img.shields.io/badge/Raspberry_Pi-OS_Bookworm-A22846?logo=raspberrypi&logoColor=white)](https://www.raspberrypi.com/)

This repository serves as a modernized update to [kinivi's hand-gesture-recognition-mediapipe](https://github.com/kinivi/hand-gesture-recognition-mediapipe) project, refactored to work with modern libraries and fully optimized for single-board computers like the Raspberry Pi.

---

## 📋 Requirements & System Setup

### 1. OS & Python Version
* **Recommended OS:** Raspberry Pi OS (64-bit) Bookworm *(Available under the "Legacy" OS section in Raspberry Pi Imager)*.
* **Python Version:** `Python 3.11.x` *(Both `3.11.15` or the default system version `3.11.2` work stably)*.

### 2. System Dependencies
Before installing Python packages, you must install the following system-level drivers for OpenCV and audio support via the terminal:

```bash
sudo apt update && sudo apt install -y \
  libgl1-mesa-glx \
  libglib2.0-0 \
  libportaudio2
🚀 Quick Start (Demo)To run the demo using your webcam, execute:Bashpython app.py
Optional ArgumentsYou can customize the camera parameters and confidence thresholds using the following flags:ArgumentDescriptionDefault--deviceSpecifying the camera device number0--widthWidth at the time of camera capture960--heightHeight at the time of camera capture540--use_static_image_modeWhether to use static_image_mode option for MediaPipeUnspecified--min_detection_confidenceDetection confidence threshold0.5--min_tracking_confidenceTracking confidence threshold0.5Example:Bashpython app.py --device 0 --width 1280 --height 720
📂 Directory StructurePlaintext.
├── app.py
├── keypoint_classification.ipynb
├── point_history_classification.ipynb
├── model/
│   ├── keypoint_classifier/
│   │   ├── keypoint.csv
│   │   ├── keypoint_classifier.hdf5
│   │   ├── keypoint_classifier.py
│   │   ├── keypoint_classifier.tflite
│   │   └── keypoint_classifier_label.csv
│   └── point_history_classifier/
│       ├── point_history.csv
│       ├── point_history_classifier.hdf5
│       ├── point_history_classifier.py
│       ├── point_history_classifier.tflite
│       └── point_history_classifier_label.csv
└── utils/
    └── cvfpscalc.py
File Componentsapp.py: The main sample program for inference. It also handles custom training data collection (keypoints & fingertip history).keypoint_classification.ipynb: Model training script for static hand signs.point_history_classification.ipynb: Model training script for dynamic finger gestures.model/: Contains training datasets (.csv), labels, saved weights, and standalone lightweight inference scripts (.py/.tflite).utils/cvfpscalc.py: Utility module for real-time FPS measurement.🏋️ Training Custom ModelsYou can easily modify existing data, inject new classes, and retrain both models to fit your specific needs.🛑 1. Hand Sign Recognition TrainingA. Learning Data CollectionPress k to enter key point logging mode (displayed on screen as MODE:Logging Key Point).Press any number from 0 to 9 to log the coordinates directly into model/keypoint_classifier/keypoint.csv.Format: 1st column = Pressed number (Class ID); subsequent columns = Preprocessed coordinates.Coordinate Preprocessing Pipeline:Default pre-loaded classes: 0 (Open Hand), 1 (Closed Hand), 2 (Pointing).B. Model TrainingOpen keypoint_classification.ipynb in Jupyter Notebook and execute the cells sequentially.If changing the number of target classes, adjust NUM_CLASSES = <your_count> and edit the respective labels inside model/keypoint_classifier/keypoint_classifier_label.csv.Network Architecture Overview:☝️ 2. Finger Gesture Recognition TrainingA. Learning Data CollectionPress h to enter fingertip history logging mode (displayed as MODE:Logging Point History).Press any number from 0 to 9 to record sequential data into model/point_history_classifier/point_history.csv.Format: 1st column = Pressed number (Class ID); subsequent columns = Coordinate displacement history.Default pre-loaded classes: 0 (Stationary), 1 (Clockwise), 2 (Counter-clockwise), 4 (Moving).B. Model TrainingOpen point_history_classification.ipynb in Jupyter Notebook and execute all cells.Remember to update NUM_CLASSES and labels in model/point_history_classifier/point_history_classifier_label.csv if customizing classes.Network Topologies (Dense vs LSTM):⚠️ Note: To leverage the recurrent LSTM model structure, set use_lstm = True within the training script.🤝 References, Authors & CreditsMediaPipe FrameworkOriginal Project Architect: Kazuhito TakahashiTranslation & Refactoring Lead: Nikita Kiselov (kinivi)📄 LicenseThis repository is open-sourced under the Apache v2 License.
### Các điểm thay đổi chính theo chuẩn GitHub:
1. **Badges**: Thêm huy hiệu (badges) trực quan ở ngay đầu dự án để người xem biết ngay Repo dùng Python mấy, OS gì và License nào.
2. **Loại bỏ lặp lại (Duplication Removal)**: Phần OS & Thư viện hệ thống của bạn bị lặp lại 2 lần (ở đầu và ở giữa), mình đã gộp sạch sẽ vào mục `Requirements & System Setup`.
3. **Thanh phân tách và Emoji**: Thêm các icon trực quan sinh động như 📋, 🚀, 🏋️ giúp phân tách các module lớn, tăng trải nghiệm đọc.
4. **Cấu trúc cây thư mục (Clean Tree)**: Chuyển cấu trúc thư mục từ dạng thẻ `<pre>
