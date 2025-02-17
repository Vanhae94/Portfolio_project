from app import create_app, db
from app.models import standardLog, NormalLog
import cv2
from ultralytics import YOLO
from datetime import datetime, timedelta
import logging

app = create_app()  # 앱을 이곳에서 한 번만 초기화

# 카메라 설정
cap_0 = cv2.VideoCapture(0)  # 0번 카메라
cap_1 = cv2.VideoCapture(1)  # 1번 카메라
        
# 해상도 설정
cap_0.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap_0.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap_1.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap_1.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Yolo 모델 로드
model = YOLO('./app/models/best250.pt')

def standard_frame():
    last_saved_time = datetime.now()  # 마지막 저장 시간을 추적

    while True:
        success_0, frame_0 = cap_0.read()  # 0번 카메라 프레임 읽기
        success_1, frame_1 = cap_1.read()  # 1번 카메라 프레임 읽기

        if not success_0 or not success_1:
            logging.error("웹캠에서 영상을 가져올 수 없습니다.")
            break

        # YOLO 모델로 객체 탐지 수행
        results_0 = model(frame_0)
        results_1 = model(frame_1)

        # 탐지된 객체를 프레임에 표시
        annotated_frame_0 = results_0[0].plot()
        annotated_frame_1 = results_1[0].plot()

        # 객체 탐지 결과 분석
        standard_detected = False
        normal_detected = True 

        def analyze_results(results):
            nonlocal standard_detected, normal_detected
            for result in results[0].boxes:
                class_id = int(result.cls[0])  # 클래스 ID
                confidence = result.conf[0]  # 신뢰도

                # 디버깅용 로그 추가
                logging.debug(f"Class ID: {class_id}, Confidence: {confidence}")

                if confidence > 0.5:
                    if "standard" in model.names[class_id].lower():
                        standard_detected = True  
                        normal_detected = False  

        analyze_results(results_0)
        analyze_results(results_1)

        # 3초마다 데이터베이스에 저장
        if datetime.now() - last_saved_time >= timedelta(seconds=3):
            try:
                with app.app_context():
                    if standard_detected:
                        logging.debug(f"Saving standard with timestamp {datetime.now()}")
                        db.session.add(standardLog(timestamp=datetime.now()))
                    elif normal_detected:
                        logging.debug(f"Saving NormalLog with timestamp {datetime.now()}")
                        db.session.add(NormalLog(timestamp=datetime.now()))

                    db.session.commit()
                    logging.debug("Database commit successful")

            except Exception as e:
                logging.error(f"Error during database commit: {e}")

            last_saved_time = datetime.now()  # 저장 시간 업데이트

        # 프레임 인코딩 후 전송 (0번과 1번을 이어붙임)
        combined_frame = cv2.hconcat([annotated_frame_0, annotated_frame_1])
        _, buffer = cv2.imencode('.jpg', combined_frame)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
