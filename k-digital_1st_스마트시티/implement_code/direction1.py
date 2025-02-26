import cv2
import numpy as np
from ultralytics import YOLO
from sort.sort import Sort  # SORT 알고리즘 사용

# YOLOv8 모델 로드 (학습된 가중치 파일 사용)
model = YOLO("bestyolo.pt")  # 학습된 가중치 파일 경로로 변경

# 영상 파일 로드
video_path = r"D:\cctv_project\video\시연영상1.mp4"  # 입력 영상 파일 경로
cap = cv2.VideoCapture(video_path)

fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# SORT 초기화
tracker = Sort()

# 구역 설정
grid_rows = 5
grid_cols = 5
cell_width = frame_width // grid_cols
cell_height = frame_height // grid_rows

# 이전 프레임의 객체 위치 저장
previous_positions = {}
future_zone_counts = []  # 미래 구역 밀집도 추적

# 구역 계산 함수
def calculate_zone(center_x, center_y):
    """객체 중심 좌표를 기반으로 구역 계산"""
    col = center_x // cell_width
    row = center_y // cell_height
    return row, col

# 영상 처리
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv8로 객체 탐지
    results = model(frame)
    detections = []

    # 탐지된 객체 정보 추출
    for result in results:
        for box, score, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
            x1, y1, x2, y2 = box.cpu().numpy().astype(int)
            score = float(score)
            detections.append([x1, y1, x2, y2, score])  # SORT 입력 형식: [x1, y1, x2, y2, score]

    # SORT로 객체 추적
    if len(detections) > 0:
        detections = np.array(detections)
    else:
        detections = np.empty((0, 5))  # 비어 있는 배열 처리
    tracks = tracker.update(detections)

    # 현재 구역별 객체 수 계산
    zone_count = np.zeros((grid_rows, grid_cols), dtype=int)
    predicted_zone_count = np.zeros((grid_rows, grid_cols), dtype=int)  # 예측 구역별 객체 수
    for track in tracks:
        x1, y1, x2, y2, track_id = track.astype(int)
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

        # 현재 구역 계산
        row, col = calculate_zone(center_x, center_y)
        if 0 <= row < grid_rows and 0 <= col < grid_cols:
            zone_count[row, col] += 1

        # 미래 위치 계산 (2초 후, FPS 기준)
        if track_id in previous_positions:
            prev_x, prev_y = previous_positions[track_id]
            dx, dy = center_x - prev_x, center_y - prev_y
            future_x = center_x + dx * (2 * fps)  # 2초 후 위치
            future_y = center_y + dy * (2 * fps)

            # 미래 구역 계산
            future_row, future_col = calculate_zone(future_x, future_y)
            if 0 <= future_row < grid_rows and 0 <= future_col < grid_cols:
                predicted_zone_count[future_row, future_col] += 1

            # 이동 방향 화살표 표시
            cv2.arrowedLine(frame, (prev_x, prev_y), (center_x, center_y), (0, 255, 255), 2)

        # 현재 위치를 이전 위치로 업데이트
        previous_positions[track_id] = (center_x, center_y)

        # 바운딩 박스와 ID 표시
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # 밀집 예상 구역 계산
    max_predicted_value = np.max(predicted_zone_count)
    predicted_dense_zone = None
    if max_predicted_value > 2: 
        max_zone_indices = np.where(predicted_zone_count == max_predicted_value)
        predicted_dense_zone = (max_zone_indices[0][0], max_zone_indices[1][0]) # 밀집 예상지역의 위치 반환 ( 행과 열의 인덱스를 반환 )

    # 4x4 구역 그리기 및 밀집 예상 구역 표시
    for row in range(grid_rows):
        for col in range(grid_cols):
            top_left = (col * cell_width, row * cell_height)
            bottom_right = ((col + 1) * cell_width, (row + 1) * cell_height)
            if (row, col) == predicted_dense_zone:
                color = (0, 0, 255)  # 빨간색: 예상 밀집 구역
                thickness = 3
            else:
                color = (200, 200, 200)  # 회색: 일반 구역
                thickness = 1
            cv2.rectangle(frame, top_left, bottom_right, color, thickness)

    # 결과 저장 및 출력
    cv2.imshow("Future Density Prediction (2s Before)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()