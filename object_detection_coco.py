import cv2
import sys

config_file = 'object_detection\\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'object_detection\\frozen_inference_graph.pb'

model = cv2.dnn_DetectionModel(frozen_model, config_file)
model.setInputSize(600, 600)
model.setInputScale(1.0 / 127.5)
model.setInputMean((127.5, 127,5, 127.5))
model.setInputSwapRB(True)

classLabels = []
file_name = 'object_detection/labels.txt'
with open(file_name, 'rt') as file:
    classLabels = file.read().rstrip('\n').split('\n')

s = 0
if len(sys.argv) > 1:
    s = sys.argv[1]

# src = cv2.VideoCapture('object_detection/video.mp4')
src = cv2.VideoCapture(s)
cv2.namedWindow("Camera Feed", cv2.WINDOW_NORMAL)
while cv2.waitKey(1) != 27:
    has_frame, frame = src.read()
    if not has_frame:
        break
    frame = cv2.resize(frame, (600, 600), interpolation = cv2.INTER_AREA)

    classIndex, confidece, bbox = model.detect(frame, confThreshold = 0.5)
    # print(classIndex)
    if len(classIndex) != 0:
        for classIdx, conf, boxes in zip(classIndex.flatten(), confidece.flatten(), bbox):
            if classIdx <= 80:
                cv2.rectangle(frame, boxes, (0,255,0), 2)
                cv2.putText(frame, classLabels[classIdx-1], (boxes[0]+10, boxes[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
    cv2.imshow("Camera Feed",frame)

src.release()
cv2.destroyWindow("Camera Feed")