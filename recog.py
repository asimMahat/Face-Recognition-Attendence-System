
# Importing necessary modules

import numpy as np
import cv2
import os
import pickle

#Using haar cascade classifier for frontal face recognition
face_cascade=cv2.CascadeClassifier('src\data\haarcascade_frontalface_alt2.xml')

# print(os.listdir('src\data'))
#Implementing the opencv recognizer
recognizer=cv2.face.LBPHFaceRecognizer_create()
recognizer.read("train1.yml")
font=cv2.FONT_HERSHEY_SIMPLEX 
color=(255,0,0)
stroke=2

labels={"person name":1}
with open("labels.pickle",'rb')as f:
    og_labels=pickle.load(f)
    labels={v:k for k,v in og_labels.items()}

#Using videocapture function to capture the video using an external USB webcam and setting into
the cap variable

cap=cv2.VideoCapture(0)
# cap.set(cv2.CV_CAP_PROP_FPS, 60)


while (True):
    ret,frame=cap.read()    
    cv2.imshow('frame',frame)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)

    for(x,y,w,h)in faces:
        print(x,y,w,h)
        # The following lines help to draw only your face neglecting other things
        roi_gray=gray[x:x+w,y:y+h]
        roi_color=frame[x:x+w,y:y+h]

        id_,conf=recognizer.predict(roi_gray)
        #giving the confidence level
        if conf>=45:
            print(id_)
            print(labels[id_])
            name=labels[id_]

            cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
        # imageFileName="7.png"
        cv2.imwrite=(imageFileName,roi_gray)
        # Drawing a rectangle in the face 
        color = (0,0,255)
        stroke=2
        width = x+w
        height = y+h
        cv2.rectangle(frame,(x,y),(width,height),color,stroke)
        
    # Showing the video with cv2 imshow function
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()













    
