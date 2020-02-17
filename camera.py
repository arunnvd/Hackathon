import cv2
import numpy as np


cap = cv2.VideoCapture(0)

classifier = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_default.xml')

while(True):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w, y+h), (255,0,0),2)

    cv2.imshow('test', frame)
    key = cv2.waitKey(10)
    if key ==32:
        break
cap.release()
cv2.destroyAllWindows()