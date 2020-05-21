# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 15:16:51 2020

@author: SACHUU
"""

import Capture as cap
import os
import cv2
import pickle
import csv



def calculate_overall_emotion(emotion_list):
    
    angry = 0
    disgusted = 0
    fearful = 0
    happy = 0
    sad = 0
    surprised = 0
    
    name = face_d()
    
    l = len(emotion_list)
    for i in range(l):
        if emotion_list[i] == "Angry":
            angry = angry + 1
        elif emotion_list[i] == "Disgusted":
            disgusted = disgusted + 1
        elif emotion_list[i] == "Fearful":
            fearful = fearful + 1
        elif emotion_list[i] == "Happy":
            happy = happy + 1
        elif emotion_list[i] == "Sad":
            sad = sad + 1
        elif emotion_list[i] == "Surprised":
            surprised = surprised + 1
        
    total = angry + disgusted + fearful + happy + sad + surprised
    
    print("Angry: {} \nDisgusted: {} \nFearful: {} \nHappy: {} \nSad: {} \nSurprised: {}".format(100*angry/total,100*disgusted/total,100*fearful/total,100*happy/total,100*sad/total,100*surprised/total))
    print("P:{}".format(100*(happy+surprised)/total))
    print("N:{}".format(100*(angry+disgusted+fearful+sad)/total))
    data_list = [name,(100*(happy+surprised)/total),(100*(angry+disgusted+fearful+sad)/total)]
    
    if not (os.path.exists("emotion.csv")):
        with open("emotion.csv",'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Name","P","N"])
            
    with open("emotion.csv", 'a+', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(data_list)
            
            
        
        
def face_d():
        cwd = os.getcwd()
        croped = os.path.join(cwd,"croped")
        face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        

        g = 0
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read("trainner.yml")
        labels = {"person_name": 1}
        with open("labels.pickle", 'rb') as f:
            og_labels = pickle.load(f)
            labels = {v:k for k,v in og_labels.items()}
            
        name = "No"
        #cap = cv2.VideoCapture(0)
        image_path = os.path.join(croped,"croped.jpg")
        img = cv2.imread(image_path)
        while True:
            #ret, img = cap.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
                
            for (x,y,w,h) in faces:
                cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = img[y:y+h, x:x+w]

                id_, conf = recognizer.predict(roi_gray)
                if conf >= 60 and conf <= 100:

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    name = labels[id_]
                    color = (255, 255, 255)
                    stroke = 2
                    #cv2.putText(img, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
                    
            #cv2.imshow('img',img)
            k = cv2.waitKey(30) & 0xff
            if k==27:
                ret, img = cap.read()
                break
            if name != "No":
                return name

        cap.release()
        cv2.destroyAllWindows()
        
#x = ['Happy','Happy','Sad','Sad','Angry','Happy','Disgusted']
#calculate_overall_emotion(x)
calculate_overall_emotion(cap.capture_image())


