<H3>proto  
  
1 : recognition  
* issue : not good performance on detecting face  

2 : mediapipe + recognition
* issue 1 improvement
* issue : when side face or far from cam, not detecting face or blur face whose registered 


3 : mediapipe + recognition(ex-frame comparison)
* issue 2 improvement
  * checking coordinates of prior frame and regard as same face if distance under 20 pixel between prior frame and present frame
