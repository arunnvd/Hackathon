import cv2
import face_recognition
import numpy as np
#from sklearn import svm


video = cv2.VideoCapture(0)
person1 = face_recognition.load_image_file("detected_faces/personId_3/bharath.jpg")
person1_encoding = face_recognition.face_encodings(person1)[0]

person2 = face_recognition.load_image_file("detected_faces/personId_2/frame_2.jpg")
person2_encoding = face_recognition.face_encodings(person2)[0]

known_face_encodings = [
    person1_encoding,
    person2_encoding
]
known_face_names = [
    "Bharath",
    "Arun"
]

while True:
    ret, frame = video.read()
    rgb_frame = frame[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings,face_encoding,tolerance=0.43)
        print(matches)

        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

        print(face_distances)

        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    
    cv2.imshow('Video', frame)

    key = cv2.waitKey(5)
    if key == 32:
        break

video.release()
cv2.destroyAllWindows()