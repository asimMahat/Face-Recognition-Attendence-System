#this script is used to train the data
#importing necessary modules
import os
import PIL
from PIL import Image
import numpy as np
import cv2
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

#finding the data for train
images = os.path.join(BASE_DIR, "faces_data")
#getting the haarcascade for frontal face
face_cascade = cv2.CascadeClassifier('src\data\haarcascade_frontalface_alt2.xml')
#creating LBPH recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

#initialization
x_train = []
y_labels = []
current_id = 0
label_ids = {}

#going through each and every dataset
for root, dirs, files in os.walk(images):
    for file in files:
        if file.endswith(".png") or (".jpg"):
            path = os.path.join(root, file)
            # print(path)
            label = os.path.basename(root).replace(" ", "-").lower()
            # print(label,path)
        if not label in label_ids:
            label_ids[label] = current_id
            current_id += 1

        id_ = label_ids[label]
        # print(label_ids)

        pil_image = Image.open(path).convert("L")

        size=(550,550)
        final_image= pil_image.resize(size,Image.ANTIALIAS)
        image_array = np.array(pil_image, "uint8")
        # print(image_array)
        faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)

        for (x, y, w, h) in faces:
            roi = image_array[y:y + h, x:x + w]
            x_train.append(roi)
            y_labels.append(id_)  

with open("labels.pickle", "wb") as f:
    pickle.dump(label_ids, f)

# print(y_labels)
# print(x_train)

#using the train method to train the dataset
recognizer.train(x_train, np.array(y_labels))

#saving the trained log 
recognizer.save("train1.yml")












