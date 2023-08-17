from ultralytics import YOLO
import cv2

model = YOLO('../Yolo-Weights/yolov8l.pt')
#yolov8n -----> n: nano
#yolov8m -----> m: medium
#yolov8l -----> l: large / slower / more detail

results = model("Images/7.jpg", show=True)
cv2.waitKey(0)