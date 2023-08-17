from ultralytics import YOLO
import cv2

cap = cv2.VideoCapture (0)  # For Webcam 0 for only one web cam , number 1,2,3 for id
cap.set(3, 1280) #prop id number / height
cap.set(4, 720) #prop id number / width

while True:
    success, img = cap.read()
    cv2.imshow ("Image", img)
    cv2.waitKey (1)
