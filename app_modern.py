import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import copy
import itertools
import csv
import time
from collections import deque
from collections import Counter

# --- Lớp tính toán FPS (Giống CvFpsCalc của dự án gốc) ---
class CvFpsCalc(object):
    def __init__(self, buffer_len=10):
        self._start_tick = cv2.getTickCount()
        self._freq = cv2.getTickFrequency()
        self._buffer_len = buffer_len
        self._times = []

    def get(self):
        current_tick = cv2.getTickCount()
        time_elapsed = (current_tick - self._start_tick) / self._freq
        self._start_tick = current_tick
        self._times.append(time_elapsed)
        if len(self._times) > self._buffer_len:
            self._times.pop(0)
        return round(1 / np.mean(self._times), 1) if np.mean(self._times) > 0 else 0.0

# --- Trình đọc danh sách nhãn ---
def load_labels(path):
    try:
        with open(path, encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            return [row[0] for row in reader]
    except Exception:
        return []

# --- Khởi tạo các nhãn ---
keypoint_classifier_labels = load_labels('model/keypoint_classifier/keypoint_classifier_label.csv')
point_history_classifier_labels = load_labels('model/point_history_classifier/point_history_classifier_label.csv')

# --- Hàm tiền xử lý dữ liệu khớp tay ---
def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y = temp_landmark_list[0][0], temp_landmark_list[0][1]
    for index, point in enumerate(temp_landmark_list):
        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = max(list(map(abs, temp_landmark_list)))
    if max_value == 0:
        max_value = 1
    return list(map(lambda n: n / max_value, temp_landmark_list))

# --- Hàm tiền xử lý lịch sử tọa độ ngón tay ---
def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]
    temp_point_history = copy.deepcopy(point_history)
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]
        temp_point_history[index][0] = (temp_point_history[index][0] - base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] - base_y) / image_height
    return list(itertools.chain.from_iterable(temp_point_history))

# --- Hàm ghi dữ liệu ra file CSV ---
def logging_csv(number, mode, landmark_list, point_history_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'model/keypoint_classifier/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    if mode == 2 and (0 <= number <= 9):
        csv_path = 'model/point_history_classifier/point_history.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *point_history_list])

# --- Hàm quản lý phím chuyển chế độ ---
def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # Phím 0 ~ 9
        number = key - 48
    if key == ord('n'):  # Chế độ bình thường
        mode = 0
    if key == ord('k'):  # Ghi Keypoint
        mode = 1
    if key == ord('h'):  # Ghi Point History
        mode = 2
    return number, mode

# --- Hàm tính toán khung bao quanh bàn tay (Bounding Box) ---
def calc_bounding_rect(landmark_list):
    landmark_array = np.array(landmark_list)
    x, y, w, h = cv2.boundingRect(landmark_array)
    return [x, y, x + w, y + h]

# --- Bộ vẽ giao diện (Bản sao hoàn chỉnh từ app.py gốc) ---
def draw_bounding_rect(image, brect):
    cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 0), 1)
    return image

def draw_info_text(image, brect, handedness, hand_sign_text, finger_gesture_text):
    cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22), (0, 0, 0), -1)
    info_text = handedness
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    cv2.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    if finger_gesture_text != "":
        cv2.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return image

def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        # Đường nối màu đen dày bọc ngoài đường màu trắng mỏng tạo viền sắc nét
        connections = [
            (2, 3), (3, 4), (5, 6), (6, 7), (7, 8),
            (9, 10), (10, 11), (11, 12), (13, 14), (14, 15), (15, 16),
            (17, 18), (18, 19), (19, 20), (0, 1), (1, 2), (2, 5),
            (5, 9), (9, 13), (13, 17), (17, 0)
        ]
        for start, end in connections:
            cv2.line(image, tuple(landmark_point[start]), tuple(landmark_point[end]), (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[start]), tuple(landmark_point[end]), (255, 255, 255), 2)

    # Vẽ các khớp ngón tay tròn rỗng viền đen nền trắng
    for index, landmark in enumerate(landmark_point):
        radius = 8 if index in [4, 8, 12, 16, 20] else 5
        cv2.circle(image, (landmark[0], landmark[1]), radius, (255, 255, 255), -1)
        cv2.circle(image, (landmark[0], landmark[1]), radius, (0, 0, 0), 1)
    return image

def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv2.circle(image, (point[0], point[1]), 1 + int(index / 2), (152, 251, 152), 2)
    return image

def draw_info(image, fps, mode, number):
    cv2.putText(image, "FPS:" + str(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, "FPS:" + str(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    mode_string = ['Logging Key Point', 'Logging Point History']
    if 1 <= mode <= 2:
        cv2.putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        if 0 <= number <= 9:
            cv2.putText(image, "NUM:" + str(number), (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return image

# --- Hàm MAIN chạy vòng lặp nhận diện ---
def main():
    # Khởi tạo mô hình TFLite Keypoint
    kp_interpreter = tf.lite.Interpreter(model_path='model/keypoint_classifier/keypoint_classifier.tflite')
    kp_interpreter.allocate_tensors()
    kp_input_details = kp_interpreter.get_input_details()
    kp_output_details = kp_interpreter.get_output_details()

    # Khởi tạo mô hình TFLite Point History
    ph_interpreter = tf.lite.Interpreter(model_path='model/point_history_classifier/point_history_classifier.tflite')
    ph_interpreter.allocate_tensors()
    ph_input_details = ph_interpreter.get_input_details()
    ph_output_details = ph_interpreter.get_output_details()

    # Các cấu hình lưu trữ lịch sử di chuyển (Point History)
    history_length = 16
    point_history = deque(maxlen=history_length)
    for _ in range(history_length):
        point_history.append([0, 0])
    finger_gesture_history = deque(maxlen=history_length)

    # Khởi tạo MediaPipe Tasks API
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    fps_calc = CvFpsCalc(buffer_len=10)
    mode = 0

    with HandLandmarker.create_from_options(options) as detector:
        while cap.isOpened():
            fps = fps_calc.get()

            # Nhận dạng sự kiện từ bàn phím
            key = cv2.waitKey(10) & 0xFF
            if key == 27 or key == ord('q'): # ESC hoặc Q để thoát ứng dụng
                break
            number, mode = select_mode(key, mode)

            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            debug_image = copy.deepcopy(frame)

            # Đọc ảnh RGB chuyển sang MediaPipe Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            
            frame_timestamp_ms = int(time.time() * 1000)
            detection_result = detector.detect_for_video(mp_image, frame_timestamp_ms)

            pointing_detected = False
            point_coord = [0, 0]

            if detection_result.hand_landmarks:
                for idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
                    handedness = detection_result.handedness[idx][0].category_name

                    # Chuyển đổi sang điểm tọa độ thực tế trên ảnh
                    landmark_list = []
                    for lm in hand_landmarks:
                        cx = min(int(lm.x * w), w - 1)
                        cy = min(int(lm.y * h), h - 1)
                        landmark_list.append([cx, cy])

                    # 1. Tính bounding box xung quanh tay
                    brect = calc_bounding_rect(landmark_list)

                    # 2. Tiền xử lý tọa độ khớp tay & lịch sử nét vẽ
                    pre_processed_landmark_list = pre_process_landmark(landmark_list)
                    pre_processed_point_history_list = pre_process_point_history(debug_image, point_history)

                    # 3. Ghi dữ liệu vào CSV (nếu ở chế độ ghi âm)
                    logging_csv(number, mode, pre_processed_landmark_list, pre_processed_point_history_list)

                    # 4. Phân loại cử chỉ tĩnh (Keypoint Classification)
                    kp_interpreter.set_tensor(kp_input_details[0]['index'], np.array([pre_processed_landmark_list], dtype=np.float32))
                    kp_interpreter.invoke()
                    kp_output = kp_interpreter.get_tensor(kp_output_details[0]['index'])
                    hand_sign_id = np.argmax(np.squeeze(kp_output))

                    # Nếu là cử chỉ chỉ tay (ID 2: Pointing), theo dõi đầu ngón trỏ (khớp 8)
                    if hand_sign_id == 2:
                        pointing_detected = True
                        point_coord = landmark_list[8]

                    # 5. Phân loại cử chỉ động dựa trên lịch sử vẽ (Point History Classification)
                    finger_gesture_id = 0
                    point_history_len = len(pre_processed_point_history_list)
                    if point_history_len == (history_length * 2):
                        ph_interpreter.set_tensor(ph_input_details[0]['index'], np.array([pre_processed_point_history_list], dtype=np.float32))
                        ph_interpreter.invoke()
                        ph_output = ph_interpreter.get_tensor(ph_output_details[0]['index'])
                        finger_gesture_id = np.argmax(np.squeeze(ph_output))

                    finger_gesture_history.append(finger_gesture_id)
                    most_common_fg_id = Counter(finger_gesture_history).most_common()

                    # Lấy tên hiển thị
                    hand_sign_text = keypoint_classifier_labels[hand_sign_id] if hand_sign_id < len(keypoint_classifier_labels) else ""
                    finger_gesture_text = point_history_classifier_labels[most_common_fg_id[0][0]] if most_common_fg_id[0][0] < len(point_history_classifier_labels) else ""

                    # 6. Vẽ các thành phần UI
                    debug_image = draw_bounding_rect(debug_image, brect)
                    debug_image = draw_landmarks(debug_image, landmark_list)
                    debug_image = draw_info_text(debug_image, brect, handedness, hand_sign_text, finger_gesture_text)

            # Cập nhật lịch sử vẽ của ngón tay
            if pointing_detected:
                point_history.append(point_coord)
            else:
                point_history.append([0, 0])

            # Vẽ nét vẽ di chuyển & thông tin chế độ góc màn hình
            debug_image = draw_point_history(debug_image, point_history)
            debug_image = draw_info(debug_image, fps, mode, number)

            # Hiển thị kết quả ra màn hình
            cv2.imshow('Hand Gesture Recognition', debug_image)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()