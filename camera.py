import cv2
import numpy as np


cap = cv2.VideoCapture(0)

classifier = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_alt2.xml')

def read_person_activity(x,y,w,h):
    print('it is working')
    return

face_rect_sum = 0
false_detection_threshold = 50
while(True):
    ret, frame = cap.read()
    face_detected = False

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(gray,scaleFactor=1.6, minNeighbors=6)

    for (x,y,w,h) in faces:
        face_detected = True
        curr_sum = 0
        cv2.rectangle(frame,(x,y),(x+w, y+h), (255,0,0),2)
        curr_sum = x + y + w + h
        print('curr = ',curr_sum,'face_sum = ',face_rect_sum,'diff = ',face_rect_sum - curr_sum)
        if face_rect_sum == 0:
            face_rect_sum = curr_sum
        elif abs(face_rect_sum - curr_sum) < false_detection_threshold:
            face_rect_sum = curr_sum
        else:
            print('False Detection ')
            face_rect_sum = 0
            

    print(face_detected, face_rect_sum)

    cv2.imshow('test', frame)





    key = cv2.waitKey(500)
    if key ==32:
        break

cap.release()
cv2.destroyAllWindows()
