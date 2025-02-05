from ultralytics import YOLO
import numpy
import cv2 as cv

model = YOLO("best50-1.pt")
cap = cv.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret :
        break

    results = model(frame)

    for result in results :
        boxes = result.boxes
        for box in boxes :
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            label = f"{model.names[cls]} {conf:.2f}"

            cv.rectangle(frame, (x1, y1),(x2, y2), (0, 255, 0), 2)
            cv.putText(frame, label, (x1,y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2 )

    cv.imshow("test", frame)

    if cv.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv.destroyAllWindows()