# app1/consumers.py - Tạo file MỚI này trong thư mục app1/

import json
import asyncio
import cv2
import mediapipe as mp
import math
import time
import base64
import numpy as np
from channels.generic.websocket import AsyncWebsocketConsumer
from asgiref.sync import sync_to_async
from send_alert import send_alert

class DrowsinessConsumer(AsyncWebsocketConsumer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize MediaPipe FaceMesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
        
        # Constants for landmarks
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]
        self.MOUTH = [61, 291, 81, 178, 13, 14, 17, 402]
        
        # Thresholds
        self.EAR_THRESHOLD = 0.2
        self.MAR_THRESHOLD = 0.75
        self.CLOSE_DURATION_THRESHOLD = 3
        self.COOLDOWN_PERIOD = 60
        
        # Timing variables
        self.start_time = None
        self.last_alert_time = 0
        self.last_yawn_alert = 0
        self.last_eye_alert = 0
        self.COOLDOWN_SECONDS = 10

    async def connect(self):
        await self.accept()
        print("WebSocket connected")
        
    async def disconnect(self, close_code):
        print(f"WebSocket disconnected with code: {close_code}")
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()

    async def receive(self, text_data):
        try:
            data = json.loads(text_data)
            if data['type'] == 'frame':
                # Decode base64 image
                image_data = data['image'].split(',')[1]
                image_bytes = base64.b64decode(image_data)
                nparr = np.frombuffer(image_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is not None:
                    # Process frame asynchronously
                    result = await self.process_frame(frame)
                    await self.send(text_data=json.dumps(result))
                
        except Exception as e:
            print(f"Error in receive: {e}")
            await self.send(text_data=json.dumps({
                'error': str(e)
            }))

    async def process_frame(self, frame):
        """Process a single frame and return the results"""
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = await sync_to_async(self.face_mesh.process)(rgb_frame)
            
            response = {
                'type': 'result',
                'face_detected': False,
                'ear_avg': 0,
                'mar': 0,
                'alert': None,
                'timestamp': time.time()
            }
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    h, w, _ = frame.shape
                    landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark]
                    
                    # Calculate metrics
                    ear_left = self.calculate_ear(landmarks, self.RIGHT_EYE)  # Flipped because camera is mirrored
                    ear_right = self.calculate_ear(landmarks, self.LEFT_EYE)
                    ear_avg = (ear_left + ear_right) / 2.0
                    mar = self.calculate_mar(landmarks, self.MOUTH)
                    
                    response.update({
                        'face_detected': True,
                        'ear_avg': round(ear_avg, 3),
                        'mar': round(mar, 3),
                        'ear_left': round(ear_left, 3),
                        'ear_right': round(ear_right, 3)
                    })
                    
                    # Check for drowsiness
                    current_time = time.time()
                    alert = self.check_drowsiness(ear_avg, mar, current_time)
                    if alert:
                        response['alert'] = alert
                        # Send alert asynchronously
                        asyncio.create_task(self.send_alert_async(alert['message']))
            
            return response
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            return {'error': str(e)}

    def euclidean(self, p1, p2):
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    def calculate_ear(self, landmarks, eye_indices):
        try:
            vertical1 = self.euclidean(landmarks[eye_indices[1]], landmarks[eye_indices[5]])
            vertical2 = self.euclidean(landmarks[eye_indices[2]], landmarks[eye_indices[4]])
            horizontal = self.euclidean(landmarks[eye_indices[0]], landmarks[eye_indices[3]])
            ear = (vertical1 + vertical2) / (2.0 * horizontal) if horizontal > 0 else 0
            return ear
        except (IndexError, ZeroDivisionError):
            return 0

    def calculate_mar(self, landmarks, mouth_indices):
        try:
            A = self.euclidean(landmarks[mouth_indices[1]], landmarks[mouth_indices[5]])
            B = self.euclidean(landmarks[mouth_indices[2]], landmarks[mouth_indices[6]])
            C = self.euclidean(landmarks[mouth_indices[3]], landmarks[mouth_indices[4]])
            D = self.euclidean(landmarks[mouth_indices[0]], landmarks[mouth_indices[7]])
            mar = (A + B + C) / (3.0 * D) if D > 0 else 0
            return mar
        except (IndexError, ZeroDivisionError):
            return 0

    def check_drowsiness(self, ear_avg, mar, current_time):
        alert = None
        
        # Check for prolonged eye closure
        if ear_avg < self.EAR_THRESHOLD:
            if self.start_time is None:
                self.start_time = current_time
            elif current_time - self.start_time > self.CLOSE_DURATION_THRESHOLD:
                if current_time - self.last_alert_time > self.COOLDOWN_PERIOD:
                    alert = {
                        'type': 'eye',
                        'message': '⚠️ Cảnh báo: Bạn đang nhắm mắt quá lâu!'
                    }
                    self.last_alert_time = current_time
                    self.last_eye_alert = current_time
        else:
            self.start_time = None
        
        # Check for yawning
        if mar > self.MAR_THRESHOLD:
            if current_time - self.last_alert_time > self.COOLDOWN_PERIOD:
                alert = {
                    'type': 'yawn',
                    'message': '😮 Cảnh báo: Có thể bạn đang ngáp.'
                }
                self.last_alert_time = current_time
                self.last_yawn_alert = current_time
        
        return alert

    async def send_alert_async(self, message):
        """Send alert asynchronously"""
        try:
            await sync_to_async(send_alert)(message)
        except Exception as e:
            print(f"Error sending alert: {e}")