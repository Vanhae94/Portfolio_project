import cv2
import torch
import numpy as np
from ultralytics import YOLO

# YOLO 모델 로드
model = YOLO('./app/models/best250.pt')  # 학습된 YOLO 모델

# # 실시간 카메라 캡처
# cap = cv2.VideoCapture(0)  # 웹캠 연결
# cap1 = cv2.VideoCapture(1)

def width_height_calculation(standard_size_cm, width_height):
    if width_height == "width" :
        cap = cv2.VideoCapture(0)
    elif width_height == "height" :
        cap = cv2.VideoCapture(1)

    max_size = 0  
    PIXEL_TO_CM = None  
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO 모델을 통해 객체 감지
        results = model(frame)

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2] 좌표

            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                width = x2 - x1  # 바운딩 박스 너비 (픽셀 단위)
                height = y2 - y1  # 바운딩 박스 높이 (픽셀 단위)

                if width_height == "width":
                    x = width
                elif width_height == "height":
                    x = height
                else:
                    return None  # 잘못된 입력 처리

                if x > max_size:  # 가장 큰 객체 기준
                    max_size = x
                    if max_size != 0:  # 0으로 나누는 오류 방지
                        PIXEL_TO_CM = standard_size_cm / max_size

        break
    
    cap.release()
    return PIXEL_TO_CM
