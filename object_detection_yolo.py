from ultralytics import YOLO
import cv2
import sys

model = YOLO('yolov8n.pt')

s=0
if len(sys.argv) > 1:
    s = sys.argv[1]
cap = cv2.VideoCapture(s)

while cv2.waitKey(1) != 27:
    has_frame, frame = cap.read()
    if not has_frame:
        break
    # if ret:
    result = model.track(frame, persist=True)

    frame_ = result[0].plot()
    cv2.imshow('Video', frame_)