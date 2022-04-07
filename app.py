from flask import Flask, render_template, Response, request
import cv2
import os
import pandas as pd
import numpy as np

from datetime import datetime
import pickle
from camera import Camera

# Using haar cascade classifier for frontal face recognition
face_cascade=cv2.CascadeClassifier('src/data/haarcascade_frontalface_alt2.xml')
 
# Creating the opencv LBPH(linear binary pattern historgram) recognizer
recognizer=cv2.face.LBPHFaceRecognizer_create()

# using the trained log obtained by training 'face_train.py'

recognizer.read("trained_logs/train1.yml")
font=cv2.FONT_HERSHEY_SIMPLEX 
color=(255,0,0)
stroke=2

labels={"person name":1}
with open("labels.pickle",'rb')as f:
    og_labels=pickle.load(f)
    labels={v:k for k,v in og_labels.items()}


app = Flask(__name__)
#rendering template

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/records')
def records():
    return render_template('records.html')

@app.route('/attendance')
def attendance():
    return render_template('attendance.html')


@app.route('/move_file')
def move_file():
    d = datetime.now()
    fileName = str(d.year)+"_"+str(d.month)+"_"+str(d.day)+".csv"
    t_date = str(d.year)+"/"+str(d.month)+"/"+str(d.day)
    if os.path.isfile(fileName):
        logDf = pd.read_csv(fileName)
    else:
        logDf = pd.DataFrame(columns=["Time","Name"])

    #getting the ss from the directory

    os.system("mv ~/Downloads/screenshot.jpg ./")
    frame = cv2.imread("screenshot.jpg")
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)
    final_face = "none"
    for(x,y,w,h)in faces:
        print(x,y,w,h)
        # The following lines help to draw only your face neglecting other things
        roi_gray=gray[x:x+w,y:y+h]
        roi_color=frame[x:x+w,y:y+h]

        id_,conf=recognizer.predict(roi_gray)
        #initializing the confidence level
        if conf>=45:
            print(id_)
            print(labels[id_])
            final_face = labels[id_]
            name=labels[id_]
            dnow = datetime.now()
                
            time= str(dnow.hour)+":"+str(dnow.minute)+":"+str(dnow.second)

            aDic = {"Time":time,"Name":final_face}
            logDf = logDf.append(aDic,ignore_index=True)

            # cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)

        imageFileName="7.png"
        cv2.imwrite=(imageFileName,roi_gray)
        # '''Drawing a rectangle in the face'''
        color = (0,0,255)
        stroke=2
        width = x+w
        height = y+h
        # cv2.rectangle(frame,(x,y),(width,height),color,stroke)
    logDf.to_csv(fileName,index=False)
    return render_template('move_file.html',my_string=final_face,my_time=time,my_date=t_date)
    
# @app.route('/get_image')
# def get_image():
#       image = flask.request.args.get('download_image')
#       print("Image aayo")


# @app.route('/new_entry')
# def new_entry():
#     return render_template('new_entry.html')

def gen():
    img = cv2.imread('lizard.jpg')
    img = cv2.resize(img, (0,0), fx=1.0, fy=1.0)
    frame = cv2.imencode('.jpg',img)[1].tobytes() # Encodes an image into a memory buffer.
    yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

#running the file

if __name__ == '__main__':
    app.run(debug=True)
