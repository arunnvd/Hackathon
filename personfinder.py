import cv2
import face_recognition
import numpy as np
from sklearn import svm
import os

SAMPLES_PATH = 'detected_faces/'
encodings = []
person_id = []
clf = svm.SVC(gamma = 'scale')


video = cv2.VideoCapture(0)

def face_train(dir_path):
    global encodings
    global person_id
    train_dir = os.listdir(dir_path);

    for person in train_dir:
        sample_pix = os.listdir(dir_path + person)

        for samples in sample_pix:
            face = face_recognition.load_image_file(dir_path + person + '/' + samples)

            face_enc = face_recognition.face_encodings(face)[0]
            encodings.append(face_enc)
            person_id.append(person)
    
    clf.fit(encodings, person_id)

def face_detect(rgb_frame):
    name = ''
    test_face_locations = face_recognition.face_locations(rgb_frame)
    num_faces = len(test_face_locations)
    for i in range(num_faces):
        test_face_encodings = face_recognition.face_encodings(rgb_frame, test_face_locations)[i]
        name = clf.predict([test_face_encodings])
    print("----------Found---------------")
    print(name)
    print("----------Found-end--------------")
    return name



face_train(SAMPLES_PATH)
while True:
    ret, frame = video.read()
    rgb_frame = frame[:, :, ::-1]

    person_id = face_detect(rgb_frame)

    if (not person_id):
        print("Nothing found")
    else:
        face_locations = face_recognition.face_locations(rgb_frame)

        for (top, right, bottom, left) in face_locations:

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, person_id[0], (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    
    cv2.imshow('Video', frame)

    key = cv2.waitKey(5)
    if key == 32:
        break

video.release()
cv2.destroyAllWindows()