import cv2
from ultralytics import YOLO

model = YOLO('./app/models/best250.pt')

def generate():
    cap = cv2.VideoCapture(0)  # 웹캠 사용

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO 모델을 사용하여 객체 감지 수행
        results = model(frame)

        # 결과에서 박스를 그리기
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # 바운딩 박스 좌표

                # 박스 그리기
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 프레임을 JPEG 형식으로 인코딩
        _, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()

        # MJPEG 스트리밍을 위한 응답 포맷
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

def generate_1():
    cap = cv2.VideoCapture(1)  # 웹캠 사용

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO 모델을 사용하여 객체 감지 수행
        results = model(frame)

        # 결과에서 박스를 그리기
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # 바운딩 박스 좌표

                # 박스 그리기
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 프레임을 JPEG 형식으로 인코딩
        _, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()

        # MJPEG 스트리밍을 위한 응답 포맷
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()