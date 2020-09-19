import cv2
import numpy as np
from tensorflow.keras.models import load_model

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = load_model('Model.h5')

cap = cv2.VideoCapture(0) 

def detect_points(image):

    img  = np.array(image)/255
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=3)
    
    pred = model.predict(img)
    face_pts = (pred*50)+100

    return face_pts

  
while True: 
      
    _,img  = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:

        roi = gray[y-70:y+h+70, x-70:x+w+70]
        org_shape = roi.shape
        roi_copy = cv2.resize(roi,(224,224))

        face_pts = detect_points(roi_copy)
        face_pts = face_pts.astype(int).reshape(-1,2)
        face_pts[:, 0] = face_pts[:, 0] * org_shape[0] / 224 + x-70
        face_pts[:, 1] = face_pts[:, 1] * org_shape[1] / 224 + y-70

        for (x,y) in zip(face_pts[:, 0], face_pts[:, 1]):
            cv2.circle(img, (x,y), 3, (0, 255, 0), -1)

        blank_img = np.zeros((480,640,1))
        for (x,y) in zip(face_pts[:, 0], face_pts[:, 1]):
            cv2.circle(blank_img, (x,y), 3, (255, 255, 225), -1)

        cv2.imshow('img', img)
        cv2.imshow('img2',blank_img)

        cv2.waitKey(5)
