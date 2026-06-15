# -*- coding: utf-8 -*-
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.lite.python.interpreter import Interpreter
import copy
import itertools
import csv
import time
import os

# Tat bot thong bao thong tin cua Tensorflow de Terminal gon gang
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- CAU HINH QUYET DINH LOAI CAMERA ---
USE_PI_CAMERA = True   # Dat True de dung Pi Camera, dat False de dung Camera USB ngoai
USB_CAMERA_INDEX = 1   # Index cua camera USB neu dung (thuong la 0, 1 hoac -1)

# --- THU VIEN PICAMERA2 (Chi load khi su dung Pi Cam) ---
if USE_PI_CAMERA:
    try:
        from picamera2 import Picamera2
    except ImportError:
        print("LOI: Khong tim thay thu vien picamera2. Hay chuyen USE_PI_CAMERA = False de dung cam USB!")

# --- KET NOI PHAN CUNG (DONG CO & CANH TAY ROBOT VIA GPIO) ---
import RPi.GPIO as GPIO
from board import SCL, SDA
import busio
from adafruit_pca9685 import PCA9685

# Khoi tao I2C va PCA9685 cho khoi dong co xe
try:
    i2c_bus = busio.I2C(SCL, SDA)
    pca = PCA9685(i2c_bus)
    pca.frequency = 50 
except Exception as e:
    print("LOI: Khong the ket noi voi PCA9685 qua I2C. Kiem tra lai day cam!")
    print(str(e))

# ================== CAU HINH CHAN GPIO BCM THEO CODE MOI ==================
SERVO_GAP = 19       # Tay gap / kep vat (Kenh 19)
SERVO_NANG_HA = 13   # Nang / ha tay (Kenh 13)

# ================== CAU HINH GOC SERVO THEO CODE MOI ==================
GOC_NANG = 90
GOC_HA = 0

GOC_MO = 40
GOC_DONG = 120

THOI_GIAN_CHO = 0.5

# Thiet lap GPIO cho tay gap
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(SERVO_GAP, GPIO.OUT)
GPIO.setup(SERVO_NANG_HA, GPIO.OUT)

pwm_gap = GPIO.PWM(SERVO_GAP, 50)
pwm_nang_ha = GPIO.PWM(SERVO_NANG_HA, 50)

pwm_gap.start(0)
pwm_nang_ha.start(0)

# Bien toan cuc quan ly trang thai cua ban
trang_thai_tay = "NANG"
trang_thai_gap = "MO"

# --- CAC HAM DIEU KHIEN TAY GAP THEO CODE MOI ---
def servo_angle(pwm, angle):
    """Dieu khien servo theo goc 0-180 do."""
    angle = max(0, min(180, angle))
    duty = 2 + (angle / 18)
    pwm.ChangeDutyCycle(duty)
    time.sleep(THOI_GIAN_CHO)
    pwm.ChangeDutyCycle(0)

def nang_tay():
    global trang_thai_tay
    print("-> Dang NANG tay...")
    servo_angle(pwm_nang_ha, GOC_NANG)
    trang_thai_tay = "NANG"

def ha_tay():
    global trang_thai_tay
    print("-> Dang HA tay...")
    servo_angle(pwm_nang_ha, GOC_HA)
    trang_thai_tay = "HA"

def mo_tay_gap():
    global trang_thai_gap
    print("-> Dang MO tay gap...")
    servo_angle(pwm_gap, GOC_MO)
    trang_thai_gap = "MO"

def dong_tay_gap():
    global trang_thai_gap
    print("-> Dang DONG/KHep tay gap...")
    servo_angle(pwm_gap, GOC_DONG)
    trang_thai_gap = "DONG"

# --- DIEU KHIEN DONG CO XE MECANUM ---
class Motor:
    def __init__(self, pwm_pin, in1_pin, in2_pin, name, inverted=False):
        self.pwm_pin = pwm_pin
        self.in1_pin = in1_pin
        self.in2_pin = in2_pin
        self.inverted = inverted

    def set_speed(self, speed):
        if self.inverted:
            speed = -speed
        duty_cycle = int(abs(speed) * 655.35)

        try:
            if speed > 0:
                pca.channels[self.in1_pin].duty_cycle = 65535
                pca.channels[self.in2_pin].duty_cycle = 0
            elif speed < 0:
                pca.channels[self.in1_pin].duty_cycle = 0
                pca.channels[self.in2_pin].duty_cycle = 65535
            else:
                pca.channels[self.in1_pin].duty_cycle = 0
                pca.channels[self.in2_pin].duty_cycle = 0
                duty_cycle = 0

            pca.channels[self.pwm_pin].duty_cycle = duty_cycle
        except Exception:
            pass

# Khoi tao 4 dong co xe Mecanum
motor_right_front = Motor(6,  5,  4, "M2+E Phai truoc", inverted=False) 
motor_left_rear = Motor(9,  11,  10, "M4+E   Trai sau  ", inverted=False)
motor_right_rear = Motor(0,  2,  1, "M1+E   Phai sau  ", inverted=False) 
motor_left_front = Motor(15, 14, 13, "M3+E Trai truoc", inverted=False)


def mecanum_drive(x, y, turn, speed_multiplier=1.0):
    K_LF, K_RF, K_LR, K_RR = 0.85, 1.0, 1.0, 0.85
    speeds = [y + x + turn, y - x - turn, y - x + turn, y + x - turn]
    max_speed = max([abs(s) for s in speeds] + [1]) # Tranh chia cho 0
    if max_speed > 100:
        speeds = [(s / max_speed) * 100 for s in speeds]
        
    motor_left_front.set_speed(speeds[0] * speed_multiplier * K_LF)
    motor_right_front.set_speed(speeds[1] * speed_multiplier * K_RF)
    motor_left_rear.set_speed(speeds[2] * speed_multiplier * K_LR)
    motor_right_rear.set_speed(speeds[3] * speed_multiplier * K_RR)

def stop_all():
    motor_left_front.set_speed(0)
    motor_right_front.set_speed(0)
    motor_left_rear.set_speed(0)
    motor_right_rear.set_speed(0)

# --- CAC HAM XU LY ANH MEDIAPIPE ---
def load_labels(path):
    try:
        with open(path, encoding='utf-8-sig') as f:
            return [row[0] for row in csv.reader(f)]
    except Exception:
        return []

keypoint_classifier_labels = load_labels('model/keypoint_classifier/keypoint_classifier_label.csv')

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y = temp_landmark_list[0][0], temp_landmark_list[0][1]
    for index, point in enumerate(temp_landmark_list):
        temp_landmark_list[index][0] -= base_x
        temp_landmark_list[index][1] -= base_y
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = max(list(map(abs, temp_landmark_list)))
    return list(map(lambda n: n / max_value if max_value != 0 else 0, temp_landmark_list))

# --- IN HUONG DAN DIEU KHIEN RA TERMINAL ---
def print_terminal_instructions():
    print("==============================================================")
    print("      HE THONG DIEU KHIEN ROBOT BANG CU CHI (CO TAY GAP)      ")
    print("==============================================================")
    print(f"  LOAI CAMERA DANG CHON: {'PI CAMERA MODULE' if USE_PI_CAMERA else 'USB CAMERA NGOAI'}")
    print("  SO DO DI DAY TAY GAP:")
    print("    - Chan GPIO 19 Pi: Tay gap / kep vat (Open/Close)")
    print("    - Chan GPIO 13 Pi: Nang / ha canh tay")
    print("==============================================================")
    print("  HUONG DAN SU DUNG (YEU CAU DONG THOI CA 2 TAY):              ")
    print("")
    print("  1. TAY TRAI: DUNG DE CHON CHE DO (MODE SELECTOR)            ")
    print("     - Xoe tay (Open) ???   --> KICH HOAT CHE DO: LAI XE")
    print("     - Nam tay (Close) ?  --> KICH HOAT CHE DO: GRIPPER / TAY GAP")
    print("     - Cac cu chi khac    --> KHOA AN TOAN (Xe dung yen)")
    print("")
    print("  2. TAY PHAI: DUNG DE DIEU KHIEN HANH DONG (ACTION)          ")
    print("     * KHI TAY TRAI DANG 'OPEN' (CHE DO LAI XE):")
    print("       + Chi 1 ngon (Pointer) ??  --> XE TIEN LEN")
    print("       + Cu chi Like (Like) ??     --> XE LUI LAI")
    print("       + Xoe tay (Open) ???        --> XE TRUOT SANG TRAI")
    print("       + Cu chi OK (OK) ??         --> XE TRUOT SANG PHAI")
    print("       + Nam tay (Close) ?        --> XE DUNG LAI")
    print("")
    print("     * KHI TAY TRAI DANG 'CLOSE' (CHE DO TAY GAP):")
    print("       + Xoe tay (Open) ???        --> MO RONG KEP GRIPPER (Goc 40)")
    print("       + Nam tay (Close) ?        --> DONG KEP GRIPPER (Goc 120)")
    print("       + Chi 1 ngon (Pointer) ??  --> NANG CANH TAY LEN (Goc 90)")
    print("       + Cu chi Like (Like) ??     --> HA CANH TAY XUONG (Goc 0)")
    print("       + Cu chi OK (OK) ??         --> RESET: NANG TAY & MO GAP")
    print("==============================================================")
    print("  Bam phim ESC hoac phim 'Q' tai cua so camera de thoat chuong trinh.")
    print("==============================================================")

# --- HAM MAIN ---
def main():
    global trang_thai_tay, trang_thai_gap

    # In huong dan va so do dieu khien ra Terminal
    print_terminal_instructions()

    # Khoi tao AI
    print("Dang tai model TFLite va MediaPipe...")
    try:
        kp_interpreter = Interpreter(model_path='model/keypoint_classifier/keypoint_classifier.tflite')
        kp_interpreter.allocate_tensors()
        kp_input_details = kp_interpreter.get_input_details()
        kp_output_details = kp_interpreter.get_output_details()
    except Exception as e:
        print("LOI: Khong the load duoc model keypoint_classifier.tflite!")
        print(str(e))
        return

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6
    )

    # Khoi dong servo ve vi tri ban dau
    print("Khoi dong servo ve vi tri ban dau...")
    nang_tay()
    mo_tay_gap()

    # --- KHOI DONG CAMERA ---
    if USE_PI_CAMERA:
        print("Dang ket noi Pi Camera module...")
        try:
            picam2 = Picamera2()
            config = picam2.create_video_configuration(main={"format": "RGB888", "size": (640, 480)})
            picam2.configure(config)
            picam2.start()
            print("Pi Camera khoi dong thanh cong!")
        except Exception as e:
            print("LOI: Khong the khoi dong Pi Camera! Hay dam bao cam dung port.")
            print(str(e))
            return
    else:
        print(f"Dang ket noi Camera USB ngoai (Index: {USB_CAMERA_INDEX})...")
        cap = cv2.VideoCapture(USB_CAMERA_INDEX)
        if not cap.isOpened() and USB_CAMERA_INDEX != 0:
            print("Khong mo duoc camera o index yeu cau, dang thu ket noi index 0...")
            cap = cv2.VideoCapture(0)
            
        if not cap.isOpened():
            print("LOI: Khong the ket noi duoc voi bat ky camera USB nao!")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print("Camera USB da ket noi thanh cong!")

    try:
        with HandLandmarker.create_from_options(options) as detector:
            while True:
                # Doc khung anh tuy thuoc vao loai camera duoc lua chon
                if USE_PI_CAMERA:
                    try:
                        frame_rgb = picam2.capture_array()
                    except Exception as e:
                        print("LOI: Mat tin hieu tu Pi Camera!")
                        break
                    frame_rgb = cv2.flip(frame_rgb, 1)
                    debug_image = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                else:
                    ret, frame_bgr = cap.read()
                    if not ret:
                        print("LOI: Mat tin hieu tu Camera USB!")
                        break
                    frame_bgr = cv2.flip(frame_bgr, 1)
                    debug_image = copy.deepcopy(frame_bgr)
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                h, w, _ = frame_rgb.shape
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                
                frame_timestamp_ms = int(time.time() * 1000)
                detection_result = detector.detect_for_video(mp_image, frame_timestamp_ms)

                left_hand_gesture = "None"
                right_hand_gesture = "None"

                if detection_result.hand_landmarks:
                    for idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
                        handedness = detection_result.handedness[idx][0].category_name
                        if handedness == "Left": handedness = "Right"
                        else: handedness = "Left"

                        landmark_list = []
                        for lm in hand_landmarks:
                            cx = min(int(lm.x * w), w - 1)
                            cy = min(int(lm.y * h), h - 1)
                            landmark_list.append([cx, cy])

                        pre_processed_landmark_list = pre_process_landmark(landmark_list)
                        kp_interpreter.set_tensor(kp_input_details[0]['index'], np.array([pre_processed_landmark_list], dtype=np.float32))
                        kp_interpreter.invoke()
                        kp_output = kp_interpreter.get_tensor(kp_output_details[0]['index'])
                        hand_sign_id = np.argmax(np.squeeze(kp_output))
                        
                        hand_sign_text = keypoint_classifier_labels[hand_sign_id] if hand_sign_id < len(keypoint_classifier_labels) else ""

                        if handedness == "Left":
                            left_hand_gesture = hand_sign_text
                        elif handedness == "Right":
                            right_hand_gesture = hand_sign_text

                        cv2.putText(debug_image, f"{handedness}: {hand_sign_text}", (landmark_list[0][0]-20, landmark_list[0][1]-20), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # --- LOGIC QUYET DINH HANH DONG ---
                base_speed = 80 

                # 1. TAY TRAI XOE (OPEN) => CHE DO LAI XE
                if left_hand_gesture == "Open":
                    cv2.putText(debug_image, "MODE: CAR DRIVE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    if right_hand_gesture == "Pointer":   # Chi 1 ngon -> Tien
                        mecanum_drive(0, base_speed, 0)
                    elif right_hand_gesture == "Like":    # Ngon cai -> Lui
                        mecanum_drive(0, -base_speed, 0)
                    elif right_hand_gesture == "Open":    # Xoe tay phai -> Truot Trai
                        mecanum_drive(-base_speed, 0, 0)
                    elif right_hand_gesture == "OK":      # Dau OK -> Truot Phai
                        mecanum_drive(base_speed, 0, 0)
                    elif right_hand_gesture == "Close":   # Nam tay -> Dung
                        stop_all()
                    else:
                        stop_all() 

                # 2. TAY TRAI NAM (CLOSE) => CHE DO CANH TAY / TAY GAP
                elif left_hand_gesture == "Close":
                    cv2.putText(debug_image, "MODE: ARM/GRIPPER", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    stop_all() # Khoa banh xe de tap trung dieu khien gap

                    # --- DIEU KHIEN CAC KHOP SERVO DUA TREN TRANG THAI GOC CO DINH ---
                    if right_hand_gesture == "Open":
                        if trang_thai_gap != "MO":
                            mo_tay_gap()
                    elif right_hand_gesture == "Close":
                        if trang_thai_gap != "DONG":
                            dong_tay_gap()
                    elif right_hand_gesture == "Pointer":
                        if trang_thai_tay != "NANG":
                            nang_tay()
                    elif right_hand_gesture == "Like":
                        if trang_thai_tay != "HA":
                            ha_tay()
                    elif right_hand_gesture == "OK":
                        # OK dung de Reset toan bo canh tay ve trang thai nang va mo rong k?p
                        if trang_thai_tay != "NANG" or trang_thai_gap != "MO":
                            nang_tay()
                            time.sleep(0.3)
                            mo_tay_gap()

                # 3. CHE DO KHOA AN TOAN (Tay trai ko xac dinh hoac ko co trong khung hinh)
                else:
                    cv2.putText(debug_image, "MODE: LOCKED (SAFE)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    stop_all()

                # Hien thi cac trang thai servo hien tai len hinh anh camera
                status_text = f"Gripper:{trang_thai_gap} | Arm_Y:{trang_thai_tay}"
                cv2.putText(debug_image, status_text, (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                cv2.imshow('Robot Vision Control', debug_image)
                
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q'):
                    break

    except KeyboardInterrupt:
        print("Da ngat boi nguoi dung tu ban phim.")
    finally:
        stop_all()
        try:
            pca.deinit()
        except NameError:
            pass
        
        # Tat chuong trinh PWM va don dep GPIO cua tay gap
        try:
            pwm_gap.stop()
            pwm_nang_ha.stop()
            GPIO.cleanup()
        except NameError:
            pass

        # Giai phong camera tuong ung
        if USE_PI_CAMERA:
            try:
                picam2.stop()
            except NameError:
                pass
        else:
            try:
                cap.release()
            except NameError:
                pass
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
