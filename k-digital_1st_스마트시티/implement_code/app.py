from flask import Flask, render_template, Response
from flask_socketio import SocketIO
import cv2
import numpy as np
from ultralytics import YOLO
from sort.sort import Sort

app = Flask(__name__)
socketio = SocketIO(app)

# YOLOv8 모델 로드
model = YOLO("bestyolo.pt")  # 학습된 가중치 파일 경로

# SORT 초기화
tracker = Sort()

# 사용자 지정 영역
start_point = None
end_point = None
selected_area_set = False
locked_area = None

# 밀집도 임계값
density_threshold = 5

# 영상 파일 로드
video_path = r"D:\cctv_project\video\KakaoTalk_20250121_120102804.mp4"  # 입력 영상 파일 경로
cap = cv2.VideoCapture(video_path)

# 히트맵 생성 함수
def generate_limited_heatmap(image, tracks, x1, y1, x2, y2, radius=50, intensity=1):
    heatmap = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
    for track in tracks:
        x1_obj, y1_obj, x2_obj, y2_obj, track_id = track.astype(int)
        center_x, center_y = (x1_obj + x2_obj) // 2, (y1_obj + y2_obj) // 2

        if x1 <= center_x <= x2 and y1 <= center_y <= y2:
            cv2.circle(heatmap, (center_x, center_y), radius, intensity, -1)

    heatmap = cv2.GaussianBlur(heatmap, (0, 0), radius)
    heatmap = np.clip(heatmap, 0, 1)
    return heatmap

# 영상 스트리밍 함수
def generate_frames():
    global start_point, end_point, selected_area_set, locked_area

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (800, 500))

        if locked_area:
            (x1_area, y1_area), (x2_area, y2_area) = locked_area
            x1_area, y1_area = min(x1_area, x2_area), min(y1_area, y2_area)
            x2_area, y2_area = max(x1_area, x2_area), max(y1_area, y2_area)

            results = model(frame)
            detections = []

            for result in results:
                for box, score, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                    x1, y1, x2, y2 = box.cpu().numpy().astype(int)
                    score = float(score)
                    detections.append([x1, y1, x2, y2, score])

            detections = np.array(detections) if len(detections) > 0 else np.empty((0, 5))
            tracks = tracker.update(detections)

            heatmap = generate_limited_heatmap(frame, tracks, x1_area, y1_area, x2_area, y2_area)
            colored_heatmap = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)

            region_density = np.sum(heatmap[y1_area:y2_area, x1_area:x2_area])
            object_count_in_area = 0

            for track in tracks:
                x1_obj, y1_obj, x2_obj, y2_obj, track_id = track.astype(int)
                center_x, center_y = (x1_obj + x2_obj) // 2, (y1_obj + y2_obj) // 2

                if x1_area <= center_x <= x2_area and y1_area <= center_y <= y2_area:
                    object_count_in_area += 1
                    cv2.rectangle(colored_heatmap, (x1_obj, y1_obj), (x2_obj, y2_obj), (255, 0, 255), 2)

            if object_count_in_area > density_threshold:
                cv2.rectangle(colored_heatmap, (x1_area, y1_area), (x2_area, y2_area), (0, 0, 255), 2)
                cv2.putText(colored_heatmap, "Warning: High Density", (x1_area + 5, y1_area - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.putText(colored_heatmap, f"Objects: {object_count_in_area}", (x1_area + 5, y1_area - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            overlay = cv2.addWeighted(frame, 0.6, colored_heatmap, 0.4, 0)
            cv2.rectangle(overlay, (x1_area, y1_area), (x2_area, y2_area), (255, 0, 0), 2)

        else:
            overlay = frame.copy()

        _, buffer = cv2.imencode('.jpg', overlay)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, threaded=True, host='0.0.0.0')
