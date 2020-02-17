import cv2
import platform

url = "http://stbimg.echostarcdn.com/danyimg/iva_trailers/221474/221474.mp4"
cap = cv2.VideoCapture(url)

while(True):
    ret, frame = cap.read()
    cv2.imshow('test',frame)
    key = cv2.waitKey(10)
    if key ==32:
        break
cap.release()
cv2.destroyAllWindows()