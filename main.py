import cv2
from ultralytics import YOLO

model = YOLO('yolo11n.pt')

video = cv2.VideoCapture('ex01.mp4')

while video.isOpened():
    check, img = video.read()
    img = cv2.resize(img, (1280, 720))
    result = model(img)
    
    for r in result:
        boxes = r.boxes
        clas = boxes.cls
        xy = boxes.xyxy.int()


        for box, cls in zip(xy, clas):
            if int(cls) ==0:
                x_min, y_min, x_max, y_max, = box.tolist()
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 3)

        cv2.imshow("Detect", img)
        cv2.waitKey(1)