
import cv2
import mediapipe as mp
import face_recognition
import numpy as np

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
# Load a sample picture and learn how to recognize it.
obama_image = face_recognition.load_image_file("obama.jpg")
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]
# Load a second sample picture and learn how to recognize it.
biden_image = face_recognition.load_image_file("biden.jpg")
biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

sumin_image = face_recognition.load_image_file("sumin.jpg")
sumin_face_encoding = face_recognition.face_encodings(sumin_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    obama_face_encoding,
    biden_face_encoding,
    sumin_face_encoding,
]
known_face_names = [
    "Barack Obama",
    "Joe Biden",
    "sumin",
]

def isClose(y1, x1, y2, x2):
    distance = (y1 - y2)**2 + (x2 - x1)**2
    return distance**0.5

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
ex_known_face_locations = []
ex_known_face_names = []
process_this_frame = True
resize_var = 4
cap = cv2.VideoCapture(0)
with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue
        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        small_image = cv2.resize(image, (0, 0), fx=1 / resize_var, fy=1 / resize_var)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        small_image.flags.writeable = False
        results = face_detection.process(small_image)
        # Draw the face detection annotations on the image.
        small_image.flags.writeable = True
        img_y, img_x, _ = small_image.shape
        rate = 20
        face_locations = []
        if results.detections:
            for detection in results.detections:
                det = detection.location_data.relative_bounding_box
                x, y, w, h = det.xmin, det.ymin, det.width, det.height
                x, y, w, h = map(int, [img_x * x, img_y * y, img_x * w, img_y * h])
                x, y = min(img_x-1, max(x, 0)), min(img_y-1, max(y, 0))
                top, right, bottom, left = y, x + w, y + h, x
                face_locations.append((top, right, bottom, left))
            face_encodings = face_recognition.face_encodings(small_image, face_locations)
            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                # # If a match was found in known_face_encodings, just use the first one.
                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_face_names[first_match_index]
                face_names.append(name)

        if ex_known_face_locations:
            for i, face_location in enumerate(face_locations):
                if face_names[i] == "Unknown":
                    for j , ex_known_face_location in enumerate(ex_known_face_locations):
                        #print(face_location)
                        #print(ex_known_face_location)
                        #print((face_location[0] + face_location[2])/2,(face_location[1] + face_location[3])/2, (ex_known_face_location[0] + ex_known_face_location[2])/2, (ex_known_face_location[1] + ex_known_face_location[3])/2)
                        #print(isClose((face_location[0] + face_location[2])/2,(face_location[1] + face_location[3])/2, (ex_known_face_location[0] + ex_known_face_location[2])/2, (ex_known_face_location[1] + ex_known_face_location[3])/2))
                        #print("---------")
                        if isClose((face_location[0] + face_location[2])/2,(face_location[1] + face_location[3])/2, (ex_known_face_location[0] + ex_known_face_location[2])/2, (ex_known_face_location[1] + ex_known_face_location[3])/2) < 20:
                            face_names[i] = ex_known_face_names[j]


        ex_known_face_locations = []
        ex_known_face_names = []

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= resize_var
            right *= resize_var
            bottom *= resize_var
            left *= resize_var
            #print(top, right, bottom, left)
            if name == "Unknown":
                roi = image[top:bottom, left:right]
                ry, rx, _ = roi.shape
                roi = cv2.resize(roi, (rx // 20, ry // 20))
                roi = cv2.resize(roi, (rx, ry), interpolation=cv2.INTER_AREA)
                image[top:bottom, left:right] = roi
                image = cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 3)
            else:
                ex_known_face_locations.append((top / resize_var, right /resize_var, bottom / resize_var, left / resize_var))
                ex_known_face_names.append(name)
                # Draw a box around the face
                cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
                # Draw a label with a name below the face
                cv2.rectangle(image, (left, bottom), (right, bottom+35), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(image, name, (left + 6, bottom + 6), font, 1.0, (255, 255, 255), 1)
        # Display the resulting image
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow('Video', image)
        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
          break
    # Release handle to the webcam
cap.release()
cv2.destroyAllWindows()







