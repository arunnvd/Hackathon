import cv2
import face_recognition
import numpy as np
from sklearn import svm
import os, signal
import time
import threading
import flask
from flask import jsonify, request
import keyboard


SAMPLES_PATH = 'detected_faces/'
TRAIN_MUTEX = False
encodings = []
person_id = []
clf = svm.SVC(gamma = 'scale')
face_train_run_once = False

stub_cam_array = [['cam1','cam2'], ['cam1'],['cam2'],['cam1'],['cam2','cam1'],['cam1','cam2'],['cam2']]


stop_thread = False




def serverHandle():
    app = flask.Flask(__name__)
    app.config["DEBUG"] = False

    @app.route('/terminate', methods=['GET'])
    def terminate():
        print("================TERMINATE=====================")
        os.kill(os.getpid(), signal.SIGTERM)

    @app.route('/userlogin', methods=['GET'])
    def home():
        if (TRAIN_MUTEX == True):
            # Trainer is busy, try again later
            return jsonify('Try_later')
        index = user_logged_in()
        if(index >= 0):
            return jsonify(stub_cam_array[index])
        else:
            return jsonify('display_all')
    app.run()


def face_train():
    global encodings
    global person_id
    global TRAIN_MUTEX
    global face_train_run_once
    train_dir = os.listdir(SAMPLES_PATH)
    print("Training Images, Please wait")

    TRAIN_MUTEX = True

    if(len(train_dir) < 2 and face_train_run_once == False):
        print("Not enough samples available.... come back later")
        if(stop_thread == False):
            threading.Timer(60, face_train).start()
        return False

    for person in train_dir:
        sample_pix = os.listdir(SAMPLES_PATH + person)

        # search for skip file in dir 
        # to avoid processing same dir again
        if person in person_id:
            print("Already processed dir, skiping dir ==> ",person)
            continue

        processed_pic_count = 0

      
        for samples in sample_pix:
            face = face_recognition.load_image_file(SAMPLES_PATH + person + '/' + samples)
            face_locations = face_recognition.face_locations(face)
            if(len(face_locations) == 1):
                face_enc = face_recognition.face_encodings(face)[0]
                encodings.append(face_enc)
                person_id.append(person)
                processed_pic_count += 1
        if(processed_pic_count == 0):
            #Not processed any images in this folder
            #ignore this entry
            print("not processed images in this folder ==> ",person)
            #return False
    
    clf.fit(encodings, person_id)
    face_train_run_once = True
    if(stop_thread == False):
        threading.Timer(60, face_train).start()
    TRAIN_MUTEX = False
    print("Face training completed")
    return True


def face_detect(rgb_frame):
    name = ''
    # run comparison only after face train classifier is ready
    if((TRAIN_MUTEX == False) and (face_train_run_once == True)):
        test_face_locations = face_recognition.face_locations(rgb_frame)
        num_faces = len(test_face_locations)
        if(num_faces == 1):
            test_face_encodings = face_recognition.face_encodings(rgb_frame, test_face_locations)[0]
            name = clf.predict([test_face_encodings])
            print("Idententified person ==> ",name)
        else:
            print("More than 1 face or no face detected, skiping this frame")
    
    else:
        print("Trainer is busy, skiping this frame")
    return name


# Trigger Face_train method to create classifier with sample pics
# this method will run once in every min after that
face_train()
threading.Timer(2, serverHandle).start()

def user_logged_in():
    run_count = 0
    video = cv2.VideoCapture(0)
    while True:
        ret, frame = video.read()
        rgb_frame = frame[:, :, ::-1]
        person_detected = False
        run_count +=1

        face_locations = face_recognition.face_locations(rgb_frame)
        if(len(face_locations) == 1):
            detected_id = face_detect(rgb_frame)

            if (not detected_id):
                print("Nothing found")
            else:
                person_detected = True
                try:
                    person_index = int(detected_id)
                except:
                    print("this is wrong name",detected_id)
                else:
                    print("person associated with cams ")
                    for camid in stub_cam_array[person_index]:
                        print(camid)
                    print("==============================")
                    video.release()
                    return person_index
        if(run_count >= 50):
            video.release()
            return -1


#while True:
keyboard.wait('esc')

print("================TERMINATE=====================")
stop_thread = True
cv2.destroyAllWindows()