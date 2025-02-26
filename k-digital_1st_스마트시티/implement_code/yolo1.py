from flask import Flask, Response, render_template_string
import cv2
from ultralytics import YOLO
import numpy as np

app = Flask(__name__)

# YOLOv8 모델 로드
model = YOLO('D:/cctv_project/runs/detect/train8/weights/best.pt')  # YOLOv8 모델 가중치 파일 로드

# 임계값 설정
THRESHOLD_COUNT = 10

# 비디오 캡처
cap1 = cv2.VideoCapture(0)  # 첫 번째 웹캠
cap2 = cv2.VideoCapture(1)  # 두 번째 웹캠

html_template = """
<!doctype html>
<html>
<head>
    <title>People Counting</title>
    <script>
        async function updateAlerts() {
            const response1 = await fetch('/alert_status_cam1');
            const response2 = await fetch('/alert_status_cam2');
            document.getElementById('alert_cam1').innerText = await response1.text();
            document.getElementById('alert_cam2').innerText = await response2.text();
        }
        setInterval(updateAlerts, 1000);
    </script>
</head>
<body>
    <h1>Real-Time People Counting</h1>
    <div>
        <h2>Camera 1</h2>
        <img src="/video_feed_cam1" width="640" height="480">
        <p id="alert_cam1" style="color: red;"></p>
    </div>
    <div>
        <h2>Camera 2</h2>
        <img src="/video_feed_cam2" width="640" height="480">
        <p id="alert_cam2" style="color: red;"></p>
    </div>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(html_template)

@app.route('/video_feed_cam1')
def video_feed_cam1():
    return Response(generate_frames(cap1, 'cam1'), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_cam2')
def video_feed_cam2():
    return Response(generate_frames(cap2, 'cam2'), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/alert_status_cam1')
def alert_status_cam1():
    global alert_message_cam1
    return alert_message_cam1

@app.route('/alert_status_cam2')
def alert_status_cam2():
    global alert_message_cam2
    return alert_message_cam2

def generate_frames(cap, cam_id):
    global alert_message_cam1, alert_message_cam2
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLOv8 추론
        results = model(frame)

        # 검출된 객체 처리
        detections = results[0].boxes.data.cpu().numpy()  # [[x1, y1, x2, y2, confidence, class], ...]

        person_count = 0
        boxes = []  # 박스 좌표 저장

        for det in detections:
            x1, y1, x2, y2, confidence, class_id = det
            if int(class_id) == 0:  # 사람 클래스 확인 및 신뢰도 조건
                boxes.append((int(x1), int(y1), int(x2), int(y2)))
                person_count += 1

        # 모든 박스의 색상 설정
        box_color = (0, 255, 0) if person_count <= THRESHOLD_COUNT else (0, 0, 255)
        for x1, y1, x2, y2 in boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

        # 경고 메시지 업데이트
        if person_count > THRESHOLD_COUNT:
            if cam_id == 'cam1':
                alert_message_cam1 = "ALERT: Too many people detected!"
            else:
                alert_message_cam2 = "ALERT: Too many people detected!"
        else:
            if cam_id == 'cam1':
                alert_message_cam1 = ""
            else:
                alert_message_cam2 = ""

        # 카운트 텍스트 추가
        cv2.putText(frame, f"Count: {person_count}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # 프레임 송출
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

if __name__ == '__main__':
    alert_message_cam1 = ""
    alert_message_cam2 = ""
    app.run(host='0.0.0.0', port=5000, debug=True)