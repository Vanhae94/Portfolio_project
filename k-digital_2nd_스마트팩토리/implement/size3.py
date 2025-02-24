import cv2
import torch
import numpy as np
from ultralytics import YOLO

# YOLO 모델 로드
model = YOLO("./k-digital_2nd_스마트팩토리/best250.pt")  # 학습된 YOLO 모델

# 실시간 카메라 캡처
cap = cv2.VideoCapture(0)  # 웹캠 연결

# 실제 기준값 입력
standard_select = input("1 (지름) / 2 (높이) 입력")
standard_size_cm = int(input("양품의 기준을 입력하세요(cm)"))
TOLERANCE_CM = int(input(" 허용 오차범위를 입력하세요 ± (cm)"))

# 호출부
if standard_select == "1" :
    # 지름 계산 함수
    def diameter_calculation() :
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

                    PIXEL_TO_CM = standard_size_cm/width

            cv2.imshow("pixel_calculation", frame)

            # ESC 키(27)를 누르면 종료
            if cv2.waitKey(1) & 0xFF == 27:
                break
        return PIXEL_TO_CM
    
    PIXEL_TO_CM = diameter_calculation()
else :
    # 높이 계산 함수
    def height_calculation() :
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

                    PIXEL_TO_CM = standard_size_cm/height

            cv2.imshow("pixel_calculation", frame)

            # ESC 키(27)를 누르면 종료
            if cv2.waitKey(1) & 0xFF == 27:
                break
        return PIXEL_TO_CM
    PIXEL_TO_CM = height_calculation()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO 모델을 통해 객체 감지
    results = model(frame)

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2] 좌표
        confidences = result.boxes.conf.cpu().numpy()  # 신뢰도
        class_ids = result.boxes.cls.cpu().numpy()  # 클래스 ID

        for box, conf, class_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = map(int, box)
            width = x2 - x1  # 바운딩 박스 너비 (픽셀 단위)
            height = y2 - y1  # 바운딩 박스 높이 (픽셀 단위)

            # ✅ 픽셀 → cm 변환 (미리 정해둔 비율 사용)
            real_size_cm = width * PIXEL_TO_CM  # 실제 크기 (cm 단위)

            # ✅ 크기 기준에 따라 양품/불량 판단
            if standard_size_cm - TOLERANCE_CM <= real_size_cm <= standard_size_cm + TOLERANCE_CM:
                status = "ACCEPTANCE "
                color = (0, 255, 0)  # 초록색 (양품)
            else:
                status = "DEFECT"
                color = (0, 0, 255)  # 빨간색 (불량)

            # 결과 표시
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{status} ({real_size_cm:.1f}cm)", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 화면 출력
    cv2.imshow("YOLOv8 Detection", frame)

    # ESC 키(27)를 누르면 종료
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
