import cv2 as cv
import mediapipe as mp
import os
from keras.models import load_model
from scipy.misc import face
import numpy as np

cap=cv.VideoCapture(0)
facedetect=cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
model=load_model('Signmodel.h5')
while cap.isOpened():
    sts,frame=cap.read()
    #frame=cv.resize(frame,(224,224),3)
    if sts:
        print(sts)
        faces=facedetect.detectMultiScale(frame,1.3,5)
        print(faces)
        for x,y,w,h in faces:
                y_pred=model.predict(frame)
                print(model.predict(frame))
                print(y_pred,"printing y_pred")
                cv.putText(frame,y_pred,(x,y-30), cv.FONT_HERSHEY_COMPLEX, 0.75, (255,0,0),1, cv.LINE_AA)
                    
    cv.imshow("SEMI",frame)
    if(cv.waitKey(5)==ord('q')):
        break
cv.destroyAllWindows()
cap.release()
