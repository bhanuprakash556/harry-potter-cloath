# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 16:47:56 2020

@author: bhanu prakash
"""

import numpy as np
import cv2
import time

cap = cv2.VideoCapture(1)

time.sleep(2)

background = 0

for i in range(30):
    ret , background = cap.read()
    

while (cap.isOpened()):
    #capture the video frame 
    ret,img = cap.read()
    if not ret:
        break
    
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    
    lower_blue = np.array([0,110,70])
    upper_blue = np.array([10,255,255])
    
    mask1 = cv2.inRange(hsv,lower_blue,upper_blue)
    
    lower_blue = np.array([170,110,70])
    upper_blue = np.array([180,255,255])
    
    
    mask2 = cv2.inRange(hsv,lower_blue,upper_blue)
    
    mask1 = mask1 + mask2
    
    mask1 = cv2.morphologyEx(mask1,cv2.MORPH_OPEN, 
                             np.ones((3,3)),iterations = 2)
    
    
    mask1 = cv2.morphologyEx(mask1,cv2.MORPH_DILATE, 
                             np.ones((3,3)),iterations = 1)
    
    mask2 = cv2.bitwise_not(mask1)
    
    res1 = cv2.bitwise_and(background,background,mask = mask1)
    res2 = cv2.bitwise_and(img,img,mask = mask2)
    res = cv2.addWeighted(res1,1,res2,1,0)
    
    #display the resulting frame
    cv2.imshow("magic",res)
    if cv2.waitKey(20) & 0xff ==ord("q"):
        break


# when everything done realease the captyre
cap.realease()
cv2.destroyAllWindows()