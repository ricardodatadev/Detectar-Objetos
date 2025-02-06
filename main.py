import cv2
from ultralytics import YOLO

model = YOLO('yolo11n.pt')

video = cv2.VideoCapture('video2.mp4')

while video.isOpened():
    check, img = video.read()
    img = cv2.resize(img, (1280, 720))
    result = model(img)