import datetime
import cv2
from app import db
from app.models import DetectionLog
import os
from datetime import datetime
from pytz import timezone

def trigger_capture(cctv_id, frame, object_count_in_area):
    base_dir = os.getcwd()  # 현재 작업 디렉토리 가져오기
    save_dir = os.path.join(base_dir, "app", "static", "images", "cctv_capture")
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.now(timezone('Asia/Seoul')).strftime("%Y%m%d_%H%M%S")
    file_name = f"{cctv_id}_{timestamp}.jpg"
    save_path = os.path.join(save_dir, file_name)

    print(f"저장 경로: {save_path}")

    try:
        result = cv2.imwrite(save_path, frame)
        if result:
            print(f"자동 캡처 성공: {save_path}")

            # 로그 저장 (상대 경로로 수정)
            relative_path = os.path.relpath(save_path, os.path.join(base_dir, 'app', 'static'))
            save_detection_log(cctv_id, object_count_in_area, relative_path)
        else:
            print(f"자동 캡처 실패: {save_path}")
    except Exception as e:
        print(f"자동 캡처 실패: {save_path}, 에러: {str(e)}")

def save_detection_log(cctv_id, density, save_path):
    try:
        # 경로 구분자를 '/'로 변경
        relative_path = save_path.replace(os.sep, '/')
        
        detection_log = DetectionLog(
            detection_time=datetime.now(timezone('Asia/Seoul')),
            cctv_id=cctv_id,
            object_count=density,
            image_url=relative_path
        )
        db.session.add(detection_log)
        db.session.commit()
        print(f"DetectionLog에 저장됨: {relative_path}")
    except Exception as e:
        print(f"DetectionLog 저장 실패: {str(e)}")

        
def generate_frames(device_index):

    # 카메라 연결 (여기서 device_index는 CCTV 장치의 번호에 해당)
    cap = cv2.VideoCapture(device_index)

    if not cap.isOpened():
        raise ValueError("CCTV 카메라를 열 수 없습니다.")

    while True:
        # 비디오에서 프레임을 읽음
        ret, frame = cap.read()
        
        if not ret:
            break  # 프레임을 더 이상 읽을 수 없으면 종료

        # 프레임을 JPEG로 인코딩
        _, jpeg = cv2.imencode('.jpg', frame)
        jpeg_bytes = jpeg.tobytes()

        # MJPEG 스트리밍을 위한 바이트 전송
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg_bytes + b'\r\n\r\n')

    # 자원 해제
    cap.release()