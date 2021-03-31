import face_recognition
import cv2
import os
import time, datetime
import numpy as np
import argparse
from skimage.metrics import structural_similarity
from .isclose import isClose
import urllib.request


def mosaic_video(recog_list, recog_images, video_route):
    known = {}

    for name in recog_list:
        known[name] = []
            
    for person_name, img_urls in zip(recog_list, recog_images):
        # print(person_name)
        # path = os.path.join(ROOT_DIR, 'known/'+person_name)
        for i, url in enumerate(img_urls):
            response = urllib.request.urlopen(url)
            face = face_recognition.load_image_file(response)
            # face = face_recognition.load_image_file(path+'/'+img_name)
            face_locations = face_recognition.face_locations(face, model='cnn')
            # If training image contains exactly one face
            if len(face_locations) == 1:
                try:
                    face_encoding = face_recognition.face_encodings(face, num_jitters=5, model="large")[0]
                    known[person_name].append(face_encoding)
                    print(person_name + "/" + str(i) +" was recognized!")
                    # Add face encoding for current image with corresponding label (name) to the training data
                except:
                    print(person_name + "/" + str(i) +" was skipped and can't be used for training cuz not detecting face")
            else:
                print(person_name + "/" + str(i) +" was skipped and can't be used for training cuz not detecting only one face ")
            # face_encodings = face_recognition.face_encodings(face)[0]

    # 두 로케이션에 대하여 비슷한 위치와 크기인지의 여부를 판단.(상대적 값으로 바꿀 필요 남아있음)



    for thresh in [0.47]:
        for match in [0.8111]:
            # Initialize some variables
            face_locations = []
            face_encodings = []
            face_names = []
            ex_known_face_locations = []
            ex_known_face_names = []

            if __name__ == '__main__':
                # input_video = args.input
                output_video = f"thresh{thresh}match{match}_"+input_video

                cap = cv2.VideoCapture(video_route)
                length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))


                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                output_movie = cv2.VideoWriter(output_video, fourcc, 29.97, (width, height))
                frame_number = 0
                ex_hist = []
                methods = {'CORREL' :cv2.HISTCMP_CORREL, 'CHISQR':cv2.HISTCMP_CHISQR,
                       'INTERSECT':cv2.HISTCMP_INTERSECT,
                       'BHATTACHARYYA':cv2.HISTCMP_BHATTACHARYYA}

                ex_frame = np.zeros([height, width, 3])
                # print(ex_frame.shape)

                # testing begin
                while cap.isOpened() and frame_number < 440:
                    success, image = cap.read()
                    frame_number += 1
                    if not success:
                        break
                    
                    if frame_number < 413:
                        continue
                    

                    # 이전 프레임과 현재 프레임간 유사도 측정
                    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                    hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
                    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
                    if len(ex_hist) == 0:
                        ex_hist = hist

                    similarity = cv2.compareHist(ex_hist, hist, methods['CORREL'])
                    print("similarity1", similarity)
                    ex_hist = hist

                    if similarity < 0.9:
                      similarity2, _ = structural_similarity(ex_frame, image, full= True, multichannel=True)
                      print("similarity2",similarity) 
                      if similarity2 < 0.7:
                        pass
                      else:
                        similarity = 1

                    ex_frame = image



                    img_raw = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    face_locations = face_recognition.face_locations(img_raw, model='cnn')
                    face_encodings = face_recognition.face_encodings(img_raw, face_locations, num_jitters=5, model="large")

                    # detection된 얼굴이 있는 경우 처리

                    face_names = []
                    for face_encoding in face_encodings:
                        for person_name in known.keys():
                            matches = face_recognition.compare_faces(known[person_name], face_encoding, tolerance=thresh)
                            name = "Unknown"

                            if matches.count(True) >= (len(known[person_name]) * match):
                                name = person_name
                                break
                        face_names.append(name)

                    tmp_locations = []
                    tmp_names = []
                    # 이전 프레임과 유사한 프레임에서 전에 얼굴을 recognition한 자리에 unknown detection이 이루어진 경우
                    # 이전 프레임에서 recognition한 얼굴로 판별한다.
                    # print(frame_number, similarity, face_names)
                    # print(frame_number, similarity, ex_known_face_names)
                    if similarity > 0.8:
                        for i, ex_known_face_location in enumerate(ex_known_face_locations):
                            for j, face_location in enumerate(face_locations):
                                if isClose(face_location, ex_known_face_location):
                                    if face_names[j] == "Unknown" or ex_known_face_names[i] != "Unknown":
                                        face_names[j] = ex_known_face_names[i]
                                        """ unknown -> unknown o
                                            unknown -> known o
                                            known -> known o
                                            known -> unknown x """
                                    # print("break!!")        
                                    break
                            else:
                                #탐지한 얼굴에 기존의 known얼굴이 없으면 추가.
                                # print("else!!")
                                tmp_locations.append(ex_known_face_location)
                                tmp_names.append(ex_known_face_names[i])

                    face_locations.extend(tmp_locations)
                    face_names.extend(tmp_names)
                    # print(tmp_names)
                    # print(frame_number, similarity, face_names)
                    # print("------------------")


                    ex_known_face_locations = face_locations[:]
                    ex_known_face_names = face_names[:]

                    for (top, right, bottom, left), name in zip(face_locations, face_names):
                        # mosaic
                        if name == "Unknown":
                            roi = img_raw[top:bottom, left:right]
                            ry, rx, _ = roi.shape
                            argResize = 30
                            if ry <= argResize or rx <= argResize:
                                continue
                            roi = cv2.resize(roi, (rx // argResize, ry // argResize))
                            roi = cv2.resize(roi, (rx, ry), interpolation=cv2.INTER_AREA)
                            img_raw[top:bottom, left:right] = roi

                        else:
                            # Draw a box around the face
                            cv2.rectangle(img_raw, (left, top), (right, bottom), (0, 0, 255), 2)
                            # Draw a label with a name below the face
                            cv2.rectangle(img_raw, (left, bottom), (right, bottom + 35), (0, 0, 255), cv2.FILLED)
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            cv2.putText(img_raw, name, (left + 6, bottom + 6), font, 1.0, (255, 255, 255), 1)

                    cv2.putText(img_raw, str(frame_number), (20,100), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (255,0,0), 1)
                    img_raw = cv2.cvtColor(img_raw, cv2.COLOR_RGB2BGR)
                    # if frame_number % 10 == 0 or frame_number == length:
                    #     print("Writing frame {} / {}".format(frame_number, length))
                    print("Writing frame {} / {}".format(frame_number, length))
                    output_movie.write(img_raw)
                    # Hit 'q' on the keyboard to quit!
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                # Release handle to the webcam
                print(f"Total time : {time.time() - startTime}")
                cap.release()
                cv2.destroyAllWindows()

