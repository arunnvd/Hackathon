import os
import cv2
import shutil
import pyrebase



cap = cv2.VideoCapture(0)
classifier = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_alt2.xml')

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

FALSE_DETECTION_THRESHOLD = 50
NEEDED_FRAMES = 5
ITRATION_THRESHOLD = 15
FRAME_CAPTURE_INTERVAL = 100 #duration between frames to be captured in msec
DETECTEDFACE_DIR_PATH = 'detected_faces/'
FBASE_CLOUD_PATH = "cam1/"

face_rect_sum = 0
false_iteration = 0
image_buff = []
person_id = 0
iteration_count = 0

def upload_to_firebase(filepath):
    firebase = pyrebase.initialize_app(config)
    storage = firebase.storage()

    complete_path_on_cloud = FBASE_CLOUD_PATH + str(person_id) + '.zip'

    storage.child(complete_path_on_cloud).put(filepath)


def clean_image_buff():
    global iteration_count
    image_buff.clear()
    iteration_count = 0
    #print('clean buff', image_buff)

def store_detected_framebuffer(roi):
    global iteration_count
    print('Saving --> ',roi.shape, ' -- ',len(image_buff))
    if(iteration_count % 7 == 0) : 
        print('saving this frame -- count ',iteration_count)
        image_buff.append(roi)
    iteration_count += 1
    return

def save_person():
    print('save Frames')
    global person_id
    frame_dir = DETECTEDFACE_DIR_PATH + 'personId_' + str(person_id)
    if (os.path.isdir(frame_dir) == False):
        os.mkdir(frame_dir)
        os.mkdir(frame_dir + '/' + str(person_id))

    for i,roi in enumerate(image_buff,start=0):
        frame_name = frame_dir +'/' + str(person_id) + '/frame_' + str(i) + '.jpg'
        cv2.imwrite(frame_name,roi)
    
    #check files saved successfully
    #if success zip the folder
    if(len(os.listdir(frame_dir + '/' +str(person_id) +'/')) >= NEEDED_FRAMES):
        print('---------ziping folder---------')
        #zipdir = frame_dir + 'ziped'
        shutil.make_archive(frame_dir,'zip',frame_dir)

    upload_to_firebase(frame_dir + '.zip')
    #clean image buffer, increment person id
    if (os.path.isdir(frame_dir) == True):
        print("removing")
       # shutil.rmtree(frame_dir)
    if (os.path.isfile(frame_dir +'.zip')):
        print("removing")
        #os.remove(frame_dir +'.zip')
    
    clean_image_buff()
    person_id += 1

while(True):
    ret, frame = cap.read()
    face_detected = False

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(gray,scaleFactor=1.4, minNeighbors=3)

    for (x,y,w,h) in faces:
        face_detected = True
        curr_sum = 0
        cv2.rectangle(frame,(x,y),(x+w, y+h), (255,0,0),2)
        curr_sum = x + y + w + h
        print('curr = ',curr_sum,'face_sum = ',face_rect_sum,'diff = ',face_rect_sum - curr_sum)
        if face_rect_sum == 0:
            face_rect_sum = curr_sum
            false_iteration = 0
            print('start detecting frames')
        elif abs(face_rect_sum - curr_sum) < FALSE_DETECTION_THRESHOLD:
            face_rect_sum = curr_sum
            false_iteration = 0
            #start storring images to buffer
            if(len(image_buff) < NEEDED_FRAMES):
                store_detected_framebuffer(gray[y:y+h, x:x+w])
            #print('store this frame to buffer')
        else:
            #print('False Detection ')
            false_iteration += 1
            if false_iteration >= ITRATION_THRESHOLD:
                #save buffer if buffered more than NEEDED_FRAMES
                if(len(image_buff) >= NEEDED_FRAMES):
                    save_person()
                print('reset buffer, and start fresh')
                face_rect_sum = 0
                #remove previous images and start fresh
                clean_image_buff()

    if(face_detected == False):
        false_iteration += 1
        if false_iteration >= ITRATION_THRESHOLD :
            #save buffer if buffered more than NEEDED_FRAMES
            if(len(image_buff) >= NEEDED_FRAMES):
                save_person()
            face_rect_sum = 0
            print('reset buffer, and start fresh due to false detection')
            #remove previous images and start fresh
            clean_image_buff()

                

    print(face_detected, face_rect_sum)

    cv2.imshow('test', frame)





    key = cv2.waitKey(FRAME_CAPTURE_INTERVAL)
    if key == 32:
        break

cap.release()
cv2.destroyAllWindows()
