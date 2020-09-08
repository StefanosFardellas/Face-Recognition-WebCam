import os
import numpy as np
import face_recognition as fr
import cv2

video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# the pictures of each person we consider as known
# (Please keep in mind that the "images\known_faces" folder is currently empty, so the paths below are not functional)
face_img1 = fr.load_image_file(os.path.abspath("images/known_faces/person_1.jpg"))
face_img2 = fr.load_image_file(os.path.abspath("images/known_faces/person_2.jpg"))

# the encodings of each known person
face_encodings1 = fr.face_encodings(face_img1)[0]
face_encodings2 = fr.face_encodings(face_img2)[0]

known_face_encodings = [
    face_encodings1,
    face_encodings2
]

# the names of each known person
known_face_names = [
    "Name 1",
    "Name 2" 
]

# draw rectangle around the detected face and show his name
def draw_face_rectangle(frame, top, right, bottom, left, name, color):
    cv2.rectangle(frame, (left - 6, top - 6), (right + 6, bottom + 6), color, 2)
    cv2.rectangle(frame, (left - 6, bottom + 6), (right + 6, bottom + 46), color, cv2.FILLED)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, name, (left, bottom + 28), font, 0.5, (255, 255, 255), 1)

while True:
    return_value, frame = video_capture.read()

    rgb_frame = frame[:, :, ::-1]

    face_locations = fr.face_locations(rgb_frame)
    face_encoding = fr.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations,face_encoding):
        matches = fr.compare_faces(known_face_encodings, face_encoding)

        face_distance = fr.face_distance(known_face_encodings, face_encoding)

        best_match_index = np.argmin(face_distance)

        name = "Unknown"
        color = (0, 0, 255)

        # if we detect a known person, then change color to green and the name to the person's name
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            color = (0, 255, 0)
        
        draw_face_rectangle(frame, top, right, bottom, left, name, color)

    cv2.imshow("Webcam Face Recognition", frame)

    # ends with pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()