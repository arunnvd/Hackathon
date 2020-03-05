import cv2
import face_recognition
import numpy as np

video = cv2.VideoCapture(0)
arun = face_recognition.load_image_file("detected_faces/personId_0/frame_0.jpg")
arun_encoding = face_recognition.face_encodings(arun)[0]

lakshmi = face_recognition.load_image_file("detected_faces/personId_3/frame_29.jpg")
lakshmi_encoding = face_recognition.face_encodings(lakshmi)[0]

known_face_encodings = [
    arun_encoding,
    lakshmi_encoding
]
known_face_names = [
    "Arun",
    "Lakshmi"
]

while True:
    ret, frame = video.read()
    rgb_frame = frame[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings,face_encoding)

        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

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