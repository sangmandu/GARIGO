import face_recognition
import cv2
import os
import time
import numpy as np

startTime = time.time()

known = {"Jae Seok": [],
         "Jong Kook": [],
         "So Min": [],
         }

ROOT_DIR = os.getcwd()
for person_name in known:
    print(person_name)
    path = os.path.join(ROOT_DIR, 'known/'+person_name)
    for img_name in os.listdir(path):
        face = face_recognition.load_image_file(path+'/'+img_name)
        face_locations = face_recognition.face_locations(face, model='cnn')

        print(img_name)
        # If training image contains exactly one face
        if len(face_locations) == 1:
            try:
                face_encoding = face_recognition.face_encodings(face, num_jitters=3, model="large")[0]
                known[person_name].append(face_encoding)
                # Add face encoding for current image with corresponding label (name) to the training data
            except:
                print(person_name + "/" + img_name + " was skipped and can't be used for training cuz not detecting face")
        else:
            print(person_name + "/" + img_name + " was skipped and can't be used for training cuz not detecting only one face ")
        # face_encodings = face_recognition.face_encodings(face)[0]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
ex_known_face_locations = []
ex_known_face_names = []
ex_unknown_face_locations = []

# 두 로케이션에 대하여 비슷한 위치와 크기인지의 여부를 판단.(상대적 값으로 바꿀 필요 남아있음)
def isClose(loc1, loc2):
    top1, right1, bottom1, left1 = loc1
    top2, right2, bottom2, left2 = loc2
    cx1, cy1, cx2, cy2 = (right1 + left1)/2, (top1 + bottom1)/2, (right2 + left2)/2, (top2 + bottom2)/2
    x1, x2, y1, y2 = right1 - left1, right2 - left2, bottom1 - top1, bottom2 - top2
    distance = (cx1 - cx2)**2 + (cy1 - cy2)**2
    distance = distance**0.5
    size_diff = abs((x1-x2) * (y1-y2))
    if distance < 10 and size_diff < 50:
        return True
    else:
        return False

if __name__ == '__main__':
    input_video = "clip.mp4"
    output_video = "output_"+input_video

    cap = cv2.VideoCapture(input_video)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_movie = cv2.VideoWriter(output_video, fourcc, 29.97, (width, height))
    frame_number = 0
    ex_hist = []
    methods = {'CORREL' :cv2.HISTCMP_CORREL, 'CHISQR':cv2.HISTCMP_CHISQR, 
           'INTERSECT':cv2.HISTCMP_INTERSECT,
           'BHATTACHARYYA':cv2.HISTCMP_BHATTACHARYYA}

    # testing begin
    while cap.isOpened() and frame_number < 1000:
        success, image = cap.read()
        frame_number += 1
        if not success:
            break

        # 이전 프레임과 현재 프레임간 유사도 측정
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        if len(ex_hist) == 0:
            ex_hist = hist
            # print("renew")

        similarity = cv2.compareHist(ex_hist, hist, methods['CORREL'])
        # print(similarity)
        ex_hist = hist

        img_raw = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(img_raw, model='cnn')
        face_encodings = face_recognition.face_encodings(img_raw, face_locations, num_jitters=3, model="large")

        # detection된 얼굴이 있는 경우 처리
        if len(face_locations) != 0:
            face_names = []
            for face_encoding in face_encodings:
                for person_name in known.keys():
                    matches = face_recognition.compare_faces(known[person_name], face_encoding, tolerance=0.4)
                    name = "Unknown"

                    print("matches.count(True): ", matches.count(True))
                    print("round(len(known[person_name]): ", round(len(known[person_name])))
                    if matches.count(True) >= round(len(known[person_name]) * 0.6):
                        name = person_name
                        break
                face_names.append(name)

            # 이전 프레임과 유사한 프레임에서 전에 얼굴을 recognition한 자리에 unknown detection이 이루어진 경우
            # 이전 프레임에서 recognition한 얼굴로 판별한다.
            if len(ex_known_face_locations) != 0 and similarity > 0.9:
                for i, face_location in enumerate(face_locations):
                    if face_names[i] == "Unknown":
                        for j, ex_known_face_location in enumerate(ex_known_face_locations):
                            if isClose(face_location, ex_known_face_location):
                                face_names[i] = ex_known_face_names[j]

            ex_known_face_locations = []
            ex_known_face_names = []
            tmp_unknown_face = ex_unknown_face_locations.copy()
            ex_unknown_face_locations = []
            Unknown_face_count = face_names.count('Unknown')

            # 이전 프레임과 다른 프레임이거나, detection한 얼굴의 개수가 같은 경우,
            # 혹은 이전에 detection한 얼굴이 없는 경우 새로 detection과 recognition진행
            if similarity < 0.9 or (similarity > 0.9 and (Unknown_face_count == len(tmp_unknown_face) or len(tmp_unknown_face) == 0)):
                # print("phase1", Unknown_face_count, len(tmp_unknown_face))
                for (top, right, bottom, left), name in zip(face_locations, face_names):
                    # mosaic
                    if name == "Unknown":
                        ex_unknown_face_locations.append((top, right, bottom , left))
                        roi = img_raw[top:bottom, left:right]
                        ry, rx, _ = roi.shape
                        if ry <= 30 or rx <= 30:
                            continue
                        roi = cv2.resize(roi, (rx // 30, ry // 30))
                        roi = cv2.resize(roi, (rx, ry), interpolation=cv2.INTER_AREA)
                        img_raw[top:bottom, left:right] = roi
                        cv2.rectangle(img_raw, (left, top), (right, bottom), (255, 0, 0), 3)

                    else:
                        ex_known_face_locations.append((top, right, bottom , left))
                        ex_known_face_names.append(name)
                        # Draw a box around the face
                        cv2.rectangle(img_raw, (left, top), (right, bottom), (0, 0, 255), 2)
                        # Draw a label with a name below the face
                        cv2.rectangle(img_raw, (left, bottom), (right, bottom + 35), (0, 0, 255), cv2.FILLED)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(img_raw, name, (left + 6, bottom + 6), font, 1.0, (255, 255, 255), 1)

            # 이전 프레임과 유사한 프레임일때, 얼굴의 개수가 달라진 경우(보완 필요!!!!) 이전 프레임의 unknown-detection을 사용한다.
            else:
                # print("phase2", Unknown_face_count, len(tmp_unknown_face))
                for (top, right, bottom, left), name in zip(face_locations, face_names):
                    # mosaic
                    if name == "Unknown":
                        continue

                    else:
                        ex_known_face_locations.append((top, right, bottom , left))
                        ex_known_face_names.append(name)
                        # Draw a box around the face
                        cv2.rectangle(img_raw, (left, top), (right, bottom), (0, 0, 255), 2)
                        # Draw a label with a name below the face
                        cv2.rectangle(img_raw, (left, bottom), (right, bottom + 35), (0, 0, 255), cv2.FILLED)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(img_raw, name, (left + 6, bottom + 6), font, 1.0, (255, 255, 255), 1)

                for (top, right, bottom, left) in tmp_unknown_face:
                    roi = img_raw[top:bottom, left:right]
                    ry, rx, _ = roi.shape
                    if ry <= 30 or rx <= 30:
                        continue
                    roi = cv2.resize(roi, (rx // 30, ry // 30))
                    roi = cv2.resize(roi, (rx, ry), interpolation=cv2.INTER_AREA)
                    img_raw[top:bottom, left:right] = roi
                    cv2.rectangle(img_raw, (left, top), (right, bottom), (255, 0, 0), 3)
                ex_unknown_face_locations = tmp_unknown_face.copy()

        # 이전 프레임과 유사한 프레임임에도 detection과 recognition아무것도 하지 못했을 때,
        # 이전 프레임의 detection과 recognition을 그대로 사용한다.
        elif len(ex_unknown_face_locations) != 0 and similarity > 0.9:
            # print("phase3",Unknown_face_count, len(tmp_unknown_face) )
            for (top, right, bottom, left), name in zip(ex_known_face_locations, ex_known_face_names):
                ex_known_face_locations.append((top, right, bottom , left))
                ex_known_face_names.append(name)
                # Draw a box around the face
                cv2.rectangle(img_raw, (left, top), (right, bottom), (0, 0, 255), 2)
                # Draw a label with a name below the face
                cv2.rectangle(img_raw, (left, bottom), (right, bottom + 35), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img_raw, name, (left + 6, bottom + 6), font, 1.0, (255, 255, 255), 1)

            for (top, right, bottom, left) in tmp_unknown_face:
                roi = img_raw[top:bottom, left:right]
                ry, rx, _ = roi.shape
                if ry <= 30 or rx <= 30:
                    continue
                roi = cv2.resize(roi, (rx // 30, ry // 30))
                roi = cv2.resize(roi, (rx, ry), interpolation=cv2.INTER_AREA)
                img_raw[top:bottom, left:right] = roi
                cv2.rectangle(img_raw, (left, top), (right, bottom), (255, 0, 0), 3)


        cv2.putText(img_raw, str(frame_number), (20,100), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (255,0,0), 1)
        img_raw = cv2.cvtColor(img_raw, cv2.COLOR_RGB2BGR)
        if frame_number % 10 == 0 or frame_number == length:
            print("Writing frame {} / {}".format(frame_number, length))
        output_movie.write(img_raw)
        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release handle to the webcam
print(f"Total time : {time.time() - startTime}")
cap.release()
cv2.destroyAllWindows()

