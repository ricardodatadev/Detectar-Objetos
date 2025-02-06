import cv2
from ultralytics import YOLO

model = YOLO('yolo11n.pt')

video = cv2.VideoCapture('video2.mp4')