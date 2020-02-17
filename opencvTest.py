import numpy as np
import cv2
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def img_load(dir):
    image_arr = []
    for file in os.listdir(dir):
        if file.endswith('png') or file.endswith('jpg') or file.endswith('jpeg'):
            rawimg = cv2.imread(os.path.join(dir, file))
            if rawimg is not None:
                image_arr.append(rawimg)
    return image_arr
                             
#img = cv2.imread('images/45bcfb5e-8b25-4d83-9c6b-61fc39b42d63.png', -1)
#classifier = cv2.CascadeClassifier('/home/pi/.local/lib/python3.5/site-packages/cv2/data/haarcascade_frontalface_default.xml')

img_arr = []
img_arr = img_load(os.path.join(BASE_DIR,'images'))

for img in img_arr:
    img_width, img_heigth = img.shape[:2]
    
    print(img_width, img_heigth)

    if img_heigth > 500 or img_width > 500:
        scale_per = 30
    else:
        scale_per = 100

    width = int(img.shape[1] * scale_per/100)
    heigth = int(img.shape[0] * scale_per/100)
    dim = (width , heigth)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    """ faces = classifier.detectMultiScale(resized)

    for box in faces:
        x,y,w,h = box
        cv2.rectangle(resized, (x,y),(x+w, y+h), (255,0,0), 2) """ 

    window = 'Test' + str(img.shape[0])
    cv2.imshow(window, resized)

key = cv2.waitKey(0)
#print("face = " , faces)

cv2.destroyAllWindows()

