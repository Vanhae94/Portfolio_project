import cv2
import numpy as np
from ultralytics import YOLO
from sort.sort import Sort  # SORT 알고리즘 사용
from app.utils import trigger_capture
import time  # time 모듈 추가

# YOLOv8 모델 로드
model = YOLO("../yolo_models/bestyolo.pt")  # 학습된 가중치 파일 경로로 변경

# SORT 초기화
tracker = Sort()

# 글로벌 변수
start_point = None
end_point = None
selected_area_set = False
locked_area = None  # 확정된 영역 저장
density_threshold = 0  # 경고를 표시할 밀집도 임계값
last_trigger_time = 0  # 마지막 trigger_capture 실행 시간
trigger_interval = 5  # trigger_capture가 실행된 후, 5초 동안 재실행 방지


def generate_frames_heatmap(device_index, cctv_id):
    global last_trigger_time  # 글로벌 변수 last_trigger_time 참조

    cap = cv2.VideoCapture(device_index)

    def generate_limited_heatmap(image, tracks, x1, y1, x2, y2, radius=50, intensity=1):
        """히트맵 생성 함수 (사용자 지정 구역 내에서만 생성)"""
        heatmap = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        for track in tracks:
            x1_obj, y1_obj, x2_obj, y2_obj, track_id = track.astype(int)
            center_x, center_y = (x1_obj + x2_obj) // 2, (y1_obj + y2_obj) // 2
            if x1 <= center_x <= x2 and y1 <= center_y <= y2:
                cv2.circle(heatmap, (center_x, center_y), radius, intensity, -1)
        heatmap = cv2.GaussianBlur(heatmap, (0, 0), radius)
        return np.clip(heatmap, 0, 1)

    def select_area(event, x, y, flags, param):
        """마우스 이벤트 핸들러: 영역 선택"""
        global start_point, end_point, selected_area_set, locked_area
        if event == cv2.EVENT_LBUTTONDOWN:  # 드래그 시작
            start_point = (x, y)
            end_point = None
            selected_area_set = False
        elif event == cv2.EVENT_MOUSEMOVE and start_point is not None:  # 드래그 중
            end_point = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:  # 드래그 종료
            end_point = (x, y)
            selected_area_set = True
            locked_area = (start_point, end_point)
        elif event == cv2.EVENT_RBUTTONDOWN:  # 우클릭으로 영역 초기화
            start_point = None
            end_point = None
            selected_area_set = False
            locked_area = None

    # OpenCV 창 설정
    cv2.namedWindow("Select Area", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Select Area", select_area)

    try:
        frame_skip = 2  # 처리할 프레임 간격
        frame_counter = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_counter += 1
            if frame_counter % frame_skip != 0:
                continue

            # 드래그 중일 때 영역 표시
            if start_point and end_point and not selected_area_set:
                cv2.rectangle(frame, start_point, end_point, (255, 0, 0), 2)

            overlay = frame.copy()

            # 확정된 영역이 있을 경우 처리
            if locked_area:
                (x1_area, y1_area), (x2_area, y2_area) = locked_area
                x1_area, y1_area = min(x1_area, x2_area), min(y1_area, y2_area)
                x2_area, y2_area = max(x1_area, x2_area), max(y1_area, y2_area)

                # YOLOv8 객체 탐지
                results = model(frame)
                detections = []

                for result in results:
                    for box, score, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                        x1, y1, x2, y2 = box.cpu().numpy().astype(int)
                        score = float(score)
                        detections.append([x1, y1, x2, y2, score])

                # SORT로 객체 추적
                detections = np.array(detections) if detections else np.empty((0, 5))
                tracks = tracker.update(detections)

                # 히트맵 생성
                heatmap = generate_limited_heatmap(frame, tracks, x1_area, y1_area, x2_area, y2_area)
                colored_heatmap = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)

                # 밀집도 및 경고 메시지
                object_count_in_area = sum(
                    x1_area <= (track[0] + track[2]) // 2 <= x2_area and y1_area <= (track[1] + track[3]) // 2 <= y2_area
                    for track in tracks
                )

                current_time = time.time()
                if object_count_in_area > density_threshold and current_time - last_trigger_time > trigger_interval:
                    trigger_capture(cctv_id, frame, object_count_in_area)
                    last_trigger_time = current_time  # trigger_capture 실행 시간 갱신

                cv2.putText(colored_heatmap, f"Objects: {object_count_in_area}", (x1_area + 5, y1_area - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                overlay = cv2.addWeighted(overlay, 0.6, colored_heatmap, 0.4, 0)

                # 사용자 지정 영역 표시
                cv2.rectangle(overlay, (x1_area, y1_area), (x2_area, y2_area), (255, 0, 0), 2)

            cv2.imshow("Select Area", overlay)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC 키를 눌렀을 때 종료
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
