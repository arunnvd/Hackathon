import numpy as np
import cv2
import os

img = cv2.imread('images/45bcfb5e-8b25-4d83-9c6b-61fc39b42d63.png', -1)
classifier = cv2.CascadeClassifier('/home/pi/.local/lib/python3.5/site-packages/cv2/data/haarcascade_frontalface_default.xml')

img_width, img_heigth = img.shape[:2]

if img_heigth > 500 and img_width > 500:
    scale_per = 30
else:
    scale_per = 100

width = int(img.shape[1] * scale_per/100)
heigth = int(img.shape[0] * scale_per/100)
dim = (width , heigth)
resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

print(resized.shape)

faces = classifier.detectMultiScale(resized)

for box in faces:
    x,y,w,h = box
    cv2.rectangle(resized, (x,y),(x+w, y+h), (255,0,0), 2) 

cv2.imshow('ArunTestWindow', resized)
#cv2.imwrite('images/resized_1.jpg' , resized)
key = cv2.waitKey(0)
print("face = " , faces)

cv2.destroyAllWindows()

