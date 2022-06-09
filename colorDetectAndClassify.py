#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2 as cv
import numpy as np


# In[2]:


class ODC:
    def __init__(self, img):
        # convert to hsv colorspace
        self.hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

        # Red color
        # lower boundary RED color range values; Hue (0 - 10)
        lower1 = np.array([0, 100, 20])
        upper1 = np.array([10, 255, 255])

        # upper boundary RED color range values; Hue (160 - 180)
        lower2 = np.array([160, 100, 20])
        upper2 = np.array([179, 255, 255])

        lower_mask = cv.inRange(self.hsv, lower1, upper1)
        upper_mask = cv.inRange(self.hsv, lower2, upper2)

        self.red_mask= lower_mask + upper_mask;

        # Blue color
        low_blue = np.array([94, 80, 2])
        high_blue = np.array([126, 255, 255])
        self.blue_mask = cv.inRange(self.hsv, low_blue, high_blue)
      
        # Green color
        low_green = np.array([25, 52, 72])
        high_green = np.array([102, 255, 255])
        self.green_mask = cv.inRange(self.hsv, low_green, high_green)
        
        # Yellow color
        low_yellow = np.array([20, 100, 100])
        high_yellow = np.array([30, 255, 255])
        self.yellow_mask = cv.inRange(self.hsv, low_yellow, high_yellow)
      
    def apply(self):
        # 1. Object Detection
        detections = []
        red_contours, _ = cv.findContours(self.red_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        for cnt in red_contours:
            # Calculate area and remove small elements
            area = cv.contourArea(cnt)
            if area > 100:
                #cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)
                x, y, w, h = cv.boundingRect(cnt)

                detections.append([x, y, w, h, "Red"])

        blue_contours, _ = cv.findContours(self.blue_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        for cnt in blue_contours:
            # Calculate area and remove small elements
            area = cv.contourArea(cnt)
            if area > 100:
                #cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)
                x, y, w, h = cv.boundingRect(cnt)

                detections.append([x, y, w, h, "Blue"])

        green_contours, _ = cv.findContours(self.green_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        for cnt in green_contours:
            # Calculate area and remove small elements
            area = cv.contourArea(cnt)
            if area > 100:
                #cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)
                x, y, w, h = cv.boundingRect(cnt)

                detections.append([x, y, w, h, "Green"])

        yellow_contours, _ = cv.findContours(self.yellow_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        for cnt in yellow_contours:
            # Calculate area and remove small elements
            area = cv.contourArea(cnt)
            if area > 100:
                #cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)
                x, y, w, h = cv.boundingRect(cnt)

                detections.append([x, y, w, h, "Yelllow"])
                
        return detections

