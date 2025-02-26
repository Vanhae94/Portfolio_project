import cv2
import numpy as np
from ultralytics import YOLO
from sort.sort import Sort  # SORT 알고리즘 사용

# YOLOv8 모델 로드 (학습된 가중치 파일 사용)
model = YOLO("bestyolo.pt")  # 학습된 가중치 파일 경로로 변경

# 영상 파일 로드
video_path = r"D:\cctv_project\video\KakaoTalk_20250121_120102804.mp4"  # 입력 영상 파일 경로
cap = cv2.VideoCapture(video_path)
output_path = "시연"  # 출력 영상 파일 경로

# 영상 저장 설정
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
output_width, output_height = 800, 500  # 출력 영상 크기
out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))

# SORT 초기화
tracker = Sort()

# 사용자 지정 영역
start_point = None
end_point = None
selected_area_set = False
locked_area = None  # 확정된 영역 저장

# 밀집도 임계값
density_threshold = 5  # 경고를 표시할 밀집도 임계값

# 히트맵 생성 함수 (사용자 지정 구역 내에서만 생성)
def generate_limited_heatmap(image, tracks, x1, y1, x2, y2, radius=50, intensity=1):
    heatmap = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
    for track in tracks:
        x1_obj, y1_obj, x2_obj, y2_obj, track_id = track.astype(int)
        center_x, center_y = (x1_obj + x2_obj) // 2, (y1_obj + y2_obj) // 2

        # 객체 중심이 지정된 영역 내에 있을 때만 히트맵 생성
        if x1 <= center_x <= x2 and y1 <= center_y <= y2:
            cv2.circle(heatmap, (center_x, center_y), radius, intensity, -1)

    heatmap = cv2.GaussianBlur(heatmap, (0, 0), radius)
    heatmap = np.clip(heatmap, 0, 1)
    return heatmap

# 마우스 콜백 함수
def select_area(event, x, y, flags, param):
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
        locked_area = (start_point, end_point)  # 확정된 영역 저장

# OpenCV 창에 마우스 콜백 연결
cv2.namedWindow("Select Area")
cv2.setMouseCallback("Select Area", select_area)

# 이전 프레임의 객체 위치 저장
previous_positions = {}

# 영상 처리
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 입력 프레임 크기 축소 (800x500)
    frame = cv2.resize(frame, (800, 500))

    # 사용자 지정 영역 표시
    if start_point and end_point and not selected_area_set:
        cv2.rectangle(frame, start_point, end_point, (255, 0, 0), 2)

    # 확정된 영역이 있다면 히트맵 생성 및 밀집도 계산
    if locked_area:
        (x1_area, y1_area), (x2_area, y2_area) = locked_area
        x1_area, y1_area = min(x1_area, x2_area), min(y1_area, y2_area)
        x2_area, y2_area = max(x1_area, x2_area), max(y1_area, y2_area)

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

        # 사용자 지정 구역 내 히트맵 생성
        heatmap = generate_limited_heatmap(frame, tracks, x1_area, y1_area, x2_area, y2_area)
        colored_heatmap = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)

        # 사용자 지정 영역 내 밀집도 및 객체 수 계산
        region_density = np.sum(heatmap[y1_area:y2_area, x1_area:x2_area])
        object_count_in_area = 0

        for track in tracks:
            x1_obj, y1_obj, x2_obj, y2_obj, track_id = track.astype(int)
            center_x, center_y = (x1_obj + x2_obj) // 2, (y1_obj + y2_obj) // 2

            if x1_area <= center_x <= x2_area and y1_area <= center_y <= y2_area:
                object_count_in_area += 1
                cv2.rectangle(colored_heatmap, (x1_obj, y1_obj), (x2_obj, y2_obj), (0, 255, 0), 2)

        # 경고 메시지 조건 수정
        if object_count_in_area > density_threshold:
            cv2.rectangle(colored_heatmap, (x1_area, y1_area), (x2_area, y2_area), (0, 0, 255), 2)
            cv2.putText(colored_heatmap, "Warning: High Density", (x1_area + 5, y1_area - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # 객체 수 정보 표시
        cv2.putText(colored_heatmap, f"Objects: {object_count_in_area}", (x1_area + 5, y1_area - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # 히트맵과 원본 프레임 결합
        overlay = cv2.addWeighted(frame, 0.6, colored_heatmap, 0.4, 0)

        # 선택된 영역 표시
        cv2.rectangle(overlay, (x1_area, y1_area), (x2_area, y2_area), (255, 0, 0), 2)

    else:
        overlay = frame.copy()

    # 결과 저장 및 출력
    out.write(overlay)
    cv2.imshow("Select Area", overlay)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # 종료 키
        break

cap.release()
out.release()
cv2.destroyAllWindows()
