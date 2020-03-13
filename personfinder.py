import cv2
import face_recognition
import numpy as np
from sklearn import svm
import os, signal
import time
import threading
import shutil
import flask
import array as arr
import zipfile
import pyrebase
from flask import jsonify, request
import keyboard
from os import path

config = {
    "apiKey": "AIzaSyCdF9n0GqIk3AHhoBUfudOp8vsiIwBxhb8",
    "authDomain": "example-image.firebaseapp.com",
    "databaseURL": "https://example-image.firebaseio.com",
    "projectId": "example-image",
    "storageBucket": "example-image.appspot.com",
    "messagingSenderId": "354350744823",
    "appId": "1:354350744823:web:39b9c35768300aad8cbff2",
    "measurementId": "G-W81CD197GY"
}
firebase = pyrebase.initialize_app(config)
storage = firebase.storage()

downloaded_CAM1_local_file_path = "cam1/"
downloaded_CAM2_local_file_path = "cam2/"


file_count_CAM1 = 0
file_count_CAM2 = 0


SAMPLES_PATH = 'detected_faces/'
TRAIN_MUTEX = False
encodings = []
person_id = []
clf = svm.SVC(gamma = 'scale')
face_train_run_once = False

download_count = 0
MaxPersons,MaxCam = (50,2)
Person_camera_map = [[0 for i in range(MaxCam)] for j in range(MaxPersons)] 
#print(Person_camera_map)
num_of_persons = 0
TempDir = 'TempDir/'
loop_continue = True

fds_count = 0



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
            return jsonify(Person_camera_map[index])
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
                    for camid in Person_camera_map[person_index]:
                        print(camid)
                    print("==============================")
                    video.release()
                    return person_index
        if(run_count >= 50):
            video.release()
            return -1


def process_update(camName, ZipFileName):
            global Person_camera_map
            global num_of_persons

            print("Input arguments to process_update: cameName --", camName, "ZipFileName -- ", ZipFileName)
            #extract camera ID from camera name /folder
            if (camName == "cam1"):
                camID = 1
                zip_file_path= downloaded_CAM1_local_file_path+ZipFileName
            else:
                if (camName == "cam2"):
                    camID = 2
                    zip_file_path= downloaded_CAM2_local_file_path+ZipFileName
                else: 
                    camID = 0

            #unzlip newly found content
            print("Unzip of : ", zip_file_path)
            #Unzip into temporary  path
            zip=zipfile.ZipFile(zip_file_path)
            zip.extractall(TempDir)

            #retrieve zip file name without extension
            unZip_Dir_name = ZipFileName[:-4]

            #call face_detect function with one of the images in upzipped directory
            unZip_Dir_path = TempDir + '/' + unZip_Dir_name
            print("UnZip directory path: ", unZip_Dir_path)
            filesList = os.listdir(unZip_Dir_path)
            
            #get rgb frame of first file in the UnZiped directory if required.
            sample_image_path = unZip_Dir_path + '/' + filesList[0]
            rgb_frame = face_recognition.load_image_file(sample_image_path)
            print("Calling face_detect functiion with rgb form of : ", sample_image_path)
            #detected_id = face_detect(rgb_frame)

            detected_id = face_detect(rgb_frame)

            if (not detected_id):
               
                print("Detection failed")
                #copy the new content to right location for training purpose with right name (num_of_persons). Delete after processing
                TempDirForTrain = SAMPLES_PATH + '/' + str(num_of_persons)
                shutil.move(unZip_Dir_path, TempDirForTrain)

                if ( True == face_train()):
                    Person_camera_map[num_of_persons][0] = camID 
                    print("person: ", num_of_persons, " assigned with came ", Person_camera_map[num_of_persons][0])
                    num_of_persons=num_of_persons+1

                shutil.rmtree(TempDirForTrain)
            else:
                try:
                    detected_index = int(detected_id)
                except:
                    print("this is wrong name",detected_id)
                else:
                    
                    #if face_detect returns success with name of the person (index), then parse zip file name, extract camera number
                    #In Person_camera_map if new camera number is not found, then append it to corresponding person row
                    print("Detected faces in: ", ZipFileName," at index: ", detected_index, "of trained data")
                    if camID in Person_camera_map[detected_index]:
                        print(camID, "camera is already part of ", detected_index, " person mapping")
                    else:
                        Person_camera_map[detected_index][1]= camID
                        print(camID, "camera is also mappped to ", detected_index, " person mapping")
            
            #remove zip file and extracted folder
            #shutil.rmtree(unZip_Dir_path)

            #print latest mapping array content
            print(np.matrix(Person_camera_map))


def check_download_start():
    print("CleanUp of CAM folders and recreate")
    shutil.rmtree(downloaded_CAM1_local_file_path)
    shutil.rmtree(downloaded_CAM2_local_file_path)
    os.makedirs(downloaded_CAM1_local_file_path)
    os.makedirs(downloaded_CAM2_local_file_path)
    check_download()

def check_download():
    
        global file_count_CAM1
        global file_count_CAM2
    
        print("In begininng of check_download func and start loop fo checking files in cam1, followed by cam2 ")
        
        while True:
            # CAM1 file download section 
            file_name_CAM1 = str(file_count_CAM1) + ".zip"
            cloud_path_CAM1 = "cam1/" + file_name_CAM1
     
            if (path.isfile(downloaded_CAM1_local_file_path+file_name_CAM1)):
                print ("CAM1 FILE  ALREADY DOWNLOADED at : ", downloaded_CAM1_local_file_path+file_name_CAM1)
                file_count_CAM1 += 1                
            else:
                print ("CAM1 FILE IS NOT AVAILABLE at: ", downloaded_CAM1_local_file_path+file_name_CAM1)
                storage.child(cloud_path_CAM1).download(downloaded_CAM1_local_file_path+file_name_CAM1)
                if (path.isfile(downloaded_CAM1_local_file_path+file_name_CAM1)):
                    print ("NOW CAM1 FILE IS DOWNLOADED at : ", downloaded_CAM1_local_file_path+file_name_CAM1)
                    process_update("cam1", file_name_CAM1)
                    file_CAM1_found=True
                    file_count_CAM1 += 1
                else:
                    print ("REQUESTED CAM1 FILE IS NOT PRESENT IN DB : ", file_name_CAM1, "cloud path: ", cloud_path_CAM1)
                    file_CAM1_found=False
                    break

        while True:
            # CAM2 file download section 
            file_name_CAM2 = str(file_count_CAM2) + ".zip"
            cloud_path_CAM2 = "cam2/" + file_name_CAM2
     
            if (path.isfile(downloaded_CAM2_local_file_path+file_name_CAM2)):
                print ("CAM2 FILE  ALREADY DOWNLOADED at : ", downloaded_CAM2_local_file_path+file_name_CAM2)
                file_count_CAM2 += 1                                    
            else:
                print ("CAM2 FILE IS NOT DOWNLOADED at: ", downloaded_CAM2_local_file_path+file_name_CAM2)
                storage.child(cloud_path_CAM2).download(downloaded_CAM2_local_file_path+file_name_CAM2)
                if (path.isfile(downloaded_CAM2_local_file_path+file_name_CAM2)):
                    print ("NOW CAM2 FILE IS DOWNLOADED at : ", downloaded_CAM2_local_file_path+file_name_CAM2)
                    process_update("cam2", file_name_CAM2)
                    file_count_CAM2 += 1                    
                    file_CAM2_found=True
                else:
                    print ("REQUESTED CAM2 FILE IS NOT PRESENT IN DB, count: ", file_name_CAM2, "cloud path: ", cloud_path_CAM2)
                    file_CAM2_found=False
                    break

        
        print("No new files avaialble, so wait for a minute and recheck. file_CAM1_found: ", file_CAM1_found, " file_CAM2_found: ", file_CAM2_found)
        threading.Timer(60, check_download).start()
        return
# Trigger Face_train method to create classifier with sample pics
# this method will run once in every min after that
check_download_start()
threading.Timer(2, serverHandle).start()
#while True:
keyboard.wait('esc')

print("================TERMINATE=====================")
stop_thread = True
cv2.destroyAllWindows()