#!/usr/bin/env python
# coding: utf-8

# In[10]:


from turtle import colormode
import cv2
from objectTracker import *
from colorDetectAndClassify import ODC


# In[15]:


# Create tracker object
tracker = EuclideanDistTracker()

cap = cv2.VideoCapture("color1.mp4")

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (700, 700), interpolation = cv2.INTER_NEAREST)
    height, width, _ = frame.shape

    # Extract Region of interest
    roi = frame[:,:]

    # 1. Object Detection
    detections = ODC(frame).apply()
    
    # 2. Object Tracking
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x, y, w, h, id, color = box_id
        cv2.putText(frame, color, (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    #cv2.imshow("roi", roi)
    cv2.imshow("Frame", frame)
    #cv2.imshow("Mask", mask)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()

