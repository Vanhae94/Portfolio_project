from app import create_app, db
from app.models import standardLog, NormalLog
import cv2
from ultralytics import YOLO
from datetime import datetime, timedelta
import logging

app = create_app()

# 카메라 설정
cap_0 = cv2.VideoCapture(0)  # 너비 측정용 카메라
cap_1 = cv2.VideoCapture(1)  # 높이 측정용 카메라

# 해상도 설정
for cap in [cap_0, cap_1]:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# YOLO 모델 로드
model = YOLO('./app/models/best250.pt')

def standard_frame_width(TOLERANCE_CM, standard_size_cm, PIXEL_TO_CM):
    last_saved_time = datetime.now()
    while True:
        success_0, frame_0 = cap_0.read()
        if not success_0 or frame_0 is None:
            logging.error("웹캠에서 영상을 가져올 수 없습니다.")
            continue
        
        results_0 = model(frame_0)
        standard_detected = False
        normal_detected = True

        for result in results_0[0].boxes:
            x1, y1, x2, y2 = map(int, result.xyxy[0].tolist())
            width = x2 - x1
            real_size_cm = width * PIXEL_TO_CM
            
            if standard_size_cm - TOLERANCE_CM <= real_size_cm <= standard_size_cm + TOLERANCE_CM:
                status, color = "ACCEPTANCE", (0, 255, 0)
            else:
                status, color = "DEFECT", (0, 0, 255)
                standard_detected, normal_detected = True, False

            cv2.rectangle(frame_0, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame_0, f"{status} ({real_size_cm:.1f}cm)", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if datetime.now() - last_saved_time >= timedelta(seconds=3):
            try:
                with app.app_context():
                    if standard_detected:
                        db.session.add(standardLog(timestamp=datetime.now()))
                    elif normal_detected:
                        db.session.add(NormalLog(timestamp=datetime.now()))
                    db.session.commit()
            except Exception as e:
                logging.error(f"DB 저장 오류: {e}")
            last_saved_time = datetime.now()
        
        _, buffer = cv2.imencode('.jpg', frame_0)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


def standard_frame_height(TOLERANCE_CM, standard_size_cm, PIXEL_TO_CM):
    last_saved_time = datetime.now()
    while True:
        success_1, frame_1 = cap_1.read()
        if not success_1 or frame_1 is None:
            logging.error("웹캠에서 영상을 가져올 수 없습니다.")
            continue
        
        results_1 = model(frame_1)
        standard_detected = False
        normal_detected = True

        for result in results_1[0].boxes:
            x1, y1, x2, y2 = map(int, result.xyxy[0].tolist())
            height = y2 - y1
            print(height)
            real_size_cm = height * PIXEL_TO_CM
            
            if standard_size_cm - TOLERANCE_CM <= real_size_cm <= standard_size_cm + TOLERANCE_CM:
                status, color = "ACCEPTANCE", (0, 255, 0)
            else:
                status, color = "DEFECT", (0, 0, 255)
                standard_detected, normal_detected = True, False

            cv2.rectangle(frame_1, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame_1, f"{status} ({real_size_cm:.1f}cm)", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if datetime.now() - last_saved_time >= timedelta(seconds=3):
            try:
                with app.app_context():
                    if standard_detected:
                        db.session.add(standardLog(timestamp=datetime.now()))
                    elif normal_detected:
                        db.session.add(NormalLog(timestamp=datetime.now()))
                    db.session.commit()
            except Exception as e:
                logging.error(f"DB 저장 오류: {e}")
            last_saved_time = datetime.now()
        
        _, buffer = cv2.imencode('.jpg', frame_1)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
