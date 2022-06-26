import cv2 as cv
import mediapipe as mp
import os
from keras.models import load_model
from scipy.misc import face
import numpy as np

cap=cv.VideoCapture(0)
facedetect=cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
model=load_model('G:\sign language projet\model.pkl')
alphab = '0123456789ABCDEFGHI$KLMNOPQRSTUVWXYZ'
mapping_letter = {}

for i,l in enumerate(alphab):
    mapping_letter[l] = i
mapping_letter = {v:k for k,v in mapping_letter.items()}
while cap.isOpened():
    sts,frame=cap.read()
    frame1=cv.resize(frame,(224,224))
    frame1 = frame1.reshape(1,224,224,3)
    if sts:
        #frame=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        faces=facedetect.detectMultiScale(frame,1.3,5)
        y_pred=model.predict(frame1)
        y1_pred = np.argmax(y_pred, axis=1)
        y_test_letters = [mapping_letter[x] for x in y1_pred]
        print(y_test_letters)
                    
    cv.imshow("SEMI",frame)
    if(cv.waitKey(5)==ord('q')):
        break
cv.destroyAllWindows()
cap.release()
