# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 11:13:20 2020

@author: SACHUU
"""

import cv2
import time
import os
import numpy as np
import emotions as em


current_folder = os.getcwd()
crop = os.path.join(current_folder,'croped\\croped.jpg')
emotion_list = []

print(p)


def capture_image():
    
    key = cv2. waitKey(1)
    webcam = cv2.VideoCapture(0)
    start_time = time.time()
    
    
    
    while True:
        try:
            check, frame = webcam.read()
            cv2.imshow("Capturing", frame)
            end_time = time.time()
            
            duration = time.strftime("%S", time.gmtime(end_time - start_time))
            key = cv2.waitKey(1)
            
            if duration == '20':
                check, img = webcam.read()
                cv2.imwrite(filename=crop, img=img)
                start_time = end_time
                emotion_list.append(em.predict_emotion(crop))
            elif key == ord('q'):
                print("Camera off.")
                print("Program ended.")
                webcam.release()
                cv2.destroyAllWindows()
                return emotion_list
                break
            
        except(KeyboardInterrupt):
            print("Turning off camera.")
            webcam.release()
            print("Camera off.")
            print("Program ended.")
            cv2.destroyAllWindows()
            break
