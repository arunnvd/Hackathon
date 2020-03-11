import cv2
import face_recognition
import numpy as np
from sklearn import svm
import os
import time
import threading


SAMPLES_PATH = 'detected_faces/'
SKIP_DIR = '.skip_dir'
TRAIN_MUTEX = False
encodings = []
person_id = []
clf = svm.SVC(gamma = 'scale')
face_train_run_once = False

stop_thread = False


video = cv2.VideoCapture(0)

def face_train():
    global encodings
    global person_id
    global TRAIN_MUTEX
    global face_train_run_once
    train_dir = os.listdir(SAMPLES_PATH)
    print("Training Images, Please wait")

    TRAIN_MUTEX = True

    if(len(train_dir) < 2):
        print("Not enough samples available.... come back later")
        return

    for person in train_dir:
        sample_pix = os.listdir(SAMPLES_PATH + person)
        do_not_process = False

        # search for skip file in dir 
        # to avoid processing same dir again
        for file_name in sample_pix:
            if(file_name == SKIP_DIR):
                do_not_process = True
                break

        if(do_not_process == False):
            for samples in sample_pix:
                face = face_recognition.load_image_file(SAMPLES_PATH + person + '/' + samples)

                face_enc = face_recognition.face_encodings(face)[0]
                encodings.append(face_enc)
                person_id.append(person)

                # create .skip_dir file to indicate folder processed already
                f = open(SAMPLES_PATH + person + '/' + SKIP_DIR, "w")
    
    clf.fit(encodings, person_id)
    face_train_run_once = True
    if(stop_thread == False):
        threading.Timer(60, face_train).start()
    TRAIN_MUTEX = False
    print("Face training completed")
    return


def face_detect(rgb_frame):
    name = ''
    # run comparison only after face train classifier is ready
    if((TRAIN_MUTEX == False) and (face_train_run_once == True)):
        test_face_locations = face_recognition.face_locations(rgb_frame)
        num_faces = len(test_face_locations)
        for i in range(num_faces):
            test_face_encodings = face_recognition.face_encodings(rgb_frame, test_face_locations)[i]
            name = clf.predict([test_face_encodings])
        print("----------Found---------------")
        print(name)
        print("----------Found-end--------------")
    
    else:
        print("Trainer is busy, skiping this frame")
    return name


# Trigger Face_train method to create classifier with sample pics
# this method will run once in every min after that
face_train()

while True:
    ret, frame = video.read()
    rgb_frame = frame[:, :, ::-1]

    detected_id = face_detect(rgb_frame)

    if (not detected_id):
        print("Nothing found")
    else:
        # Test code only - not required in actual implementation
        face_locations = face_recognition.face_locations(rgb_frame)

        for (top, right, bottom, left) in face_locations:

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, detected_id[0], (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    
    cv2.imshow('Video', frame)

    key = cv2.waitKey(500)
    if key == 32:
        break
stop_thread = True
video.release()
cv2.destroyAllWindows()