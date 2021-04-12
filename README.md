# GARIGO
#### make face of un-related people mosaic in video
#### cooperate with apeltop, lim8540

## what
* when capturing or broadcasting outside, it can be possible that people(or their face) show in the program but they are not related in this and it woulb be issue between common-person and creator
* so, this project wants to show related actor only in program or video

## static, not dynamic(real-time)
* this project can be applied to static video or image
* due to speed and performance, this is not appropriate for realtime video

## the way to make better performance
* compare with frames whose similarity is high(in this code, similarity over 0.9(or 0.8) is regard as similar)
* up-scaling
* argument of detection
  * model=cnn (<> hog)
  * num of upscaling = 2 (default 1)
* argument of recognition
  * num_jitters=5 (default 1, when number up, times up)
  * model="large" (<> small)

## face detection and recognition
https://pypi.org/project/face-recognition/

https://github.com/ageitgey/face_recognition

