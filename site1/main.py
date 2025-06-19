import cv2
import mediapipe as mp
import math
import time
from send_alert import send_alert

# Khởi tạo MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils

# Camera
cap = cv2.VideoCapture(0)

# Các chỉ số landmarks cho mắt và miệng (MediaPipe)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [61, 291, 81, 178, 13, 14, 17, 402]  # MAR

# Ngưỡng cảnh báo
EAR_THRESHOLD = 0.2
MAR_THRESHOLD = 0.75
CLOSE_DURATION_THRESHOLD = 3  # giây mắt nhắm liên tục
COOLDOWN_PERIOD = 60  # giây giữa các cảnh báo

# Biến để theo dõi thời gian cảnh báo
last_alert_time = 0
last_yawn_alert = 0
last_eye_alert = 0
COOLDOWN_SECONDS = 10  # giây giữa các cảnh báo

# Biến thời gian
start_time = None
last_alert_time = 0

# Hàm tính khoảng cách Euclidean
def euclidean(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

# Hàm tính Eye Aspect Ratio
def calculate_ear(landmarks, eye_indices):
    vertical1 = euclidean(landmarks[eye_indices[1]], landmarks[eye_indices[5]])
    vertical2 = euclidean(landmarks[eye_indices[2]], landmarks[eye_indices[4]])
    horizontal = euclidean(landmarks[eye_indices[0]], landmarks[eye_indices[3]])
    ear = (vertical1 + vertical2) / (2.0 * horizontal)
    return ear

# Hàm tính Mouth Aspect Ratio (ngáp)
def calculate_mar(landmarks, mouth_indices):
    A = euclidean(landmarks[mouth_indices[1]], landmarks[mouth_indices[5]])
    B = euclidean(landmarks[mouth_indices[2]], landmarks[mouth_indices[6]])
    C = euclidean(landmarks[mouth_indices[3]], landmarks[mouth_indices[4]])
    D = euclidean(landmarks[mouth_indices[0]], landmarks[mouth_indices[7]])
    mar = (A + B + C) / (3.0 * D)
    return mar

# Vòng lặp chính
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)
            h, w, _ = frame.shape
            landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark]

            # Vẽ điểm mắt (debug)
            for idx in LEFT_EYE + RIGHT_EYE:
                cv2.circle(frame, landmarks[idx], 2, (255, 0, 0), -1)

            # Tính toán chỉ số EAR và MAR
            ear_left = calculate_ear(landmarks, RIGHT_EYE)  # do ảnh đã flip
            ear_right = calculate_ear(landmarks, LEFT_EYE)
            ear_avg = (ear_left + ear_right) / 2.0
            mar = calculate_mar(landmarks, MOUTH)

            # Hiển thị chỉ số
            y_start = 30
            spacing = 30
            line = 0
            alert_y = y_start + spacing * line

            cv2.putText(frame, f'LEFT EAR: {ear_left:.2f}', (30, y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f'RIGHT EAR: {ear_right:.2f}', (30, y_start + spacing), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            cv2.putText(frame, f'AVG EAR: {ear_avg:.2f}', (30, y_start + spacing * 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f'MAR: {mar:.2f}', (30, y_start + spacing * 3), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 100, 255), 2)


            current_time = time.time()
            # Cảnh báo do nhắm mắt quá lâu
            if ear_avg < EAR_THRESHOLD:
                if start_time is None:
                    start_time = current_time
                elif current_time - start_time > CLOSE_DURATION_THRESHOLD:
                    if current_time - last_alert_time > COOLDOWN_PERIOD:
                        cv2.putText(frame, 'DROWSY (EYE)', (30, alert_y+ spacing * 4), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                        print("[ALERT] DROWSY (EYE)")
                        last_alert_time = current_time
            else:
                start_time = None
            
            # Cảnh báo do ngáp
            if mar > MAR_THRESHOLD:
                if current_time - last_alert_time > COOLDOWN_PERIOD:
                    cv2.putText(frame, 'DROWSY (YAWNING)', (30, alert_y + spacing * 4), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 128, 255), 3)                
                    print("[ALERT] DROWSY (YAWNING)")
                    last_alert_time = current_time

            if ear_avg < EAR_THRESHOLD and current_time - last_eye_alert > COOLDOWN_SECONDS:
                send_alert("⚠️ Cảnh báo: Bạn đang nhắm mắt quá lâu!")
                last_eye_alert = current_time

            # Ngáp
            if mar > MAR_THRESHOLD and current_time - last_yawn_alert > COOLDOWN_SECONDS:
                send_alert("😮 Cảnh báo: Có thể bạn đang ngáp.")
                last_yawn_alert = current_time

    cv2.imshow('Drowsiness Detection - EAR + MAR (MediaPipe)', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()