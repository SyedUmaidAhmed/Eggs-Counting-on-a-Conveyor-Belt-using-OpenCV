import numpy as np

#NumPy is a Python library used for working with arrays
#It also has functions for working in domain of linear algebra
#fourier transform, and matrices

import pprint

#The pprint module provides a capability to “pretty-print” arbitrary
#Python data structures in a well-formatted, Indented

import sys

#The sys module provides information about constants
#functions and methods of the Python interpreter

import datetime #DateTime Importing
import math #Maths Library for mathematical functions
import cv2 #Python Image Processing Library
import matplotlib.pyplot as plt  # Plotting Taks
from getdist import plots, MCSamples

#GetDist is a Python package for analysing Monte Carlo
#samples, including correlated samples from Markov Chain Monte Carlo (MCMC).

from PyQt5.QtWidgets import QApplication #Making Desktop Application

# global variables in Other Python Files
# Basically we used it for Making GUI in Classes Forms
from dialogs.settings_dialog import Settings

# We will use all these variables 

width = 0
height = 0
eggCount = 0
exitCounter = 0
OffsetRefLines = 50  # Adjust ths value according to your usage
ReferenceFrame = None
distance_tresh = 200
radius_min = 0
radius_max = 0
area_min = 0
area_max = 0

# Making object of classes from above file

app = QApplication(sys.argv)
set = Settings()
#sys.exit(app.exec_())

# Function to resize the frame of this Video Capturing
# resizing the frame from variable "ret,frame"
# The shape attribute for numpy arrays returns the dimensions
# of the array. If Y has n rows and m columns, then Y.shape is (n,m)
# So Y.shape[0] is n
# floor division  5//2 = 2 (Returns Only Integer Part)

# np.int0() is done to convert the corrdinates from float to integer format.


def reScaleFrame(frame, percent=75):
    width = int(frame.shape[1] * percent // 100)
    height = int(frame.shape[0] * percent // 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

#cv2.INTER_AREA method for interpolation. I
#Interpolation function’s goal is to examine neighborhoods of pixels
# and use these neighborhoods to optically increase or decrease the
# size of the image without introducing distortion


def CheckInTheArea(coordYContour, coordYEntranceLine, coordYExitLine):
    if ((coordYContour <= coordYEntranceLine) and (coordYContour >= coordYExitLine)):
        return 1
    else:
        return 0


def CheckEntranceLineCrossing(coordYContour, coordYEntranceLine):
    absDistance = abs(coordYContour - coordYEntranceLine)

    if ((coordYContour >= coordYEntranceLine) and (absDistance <= 3)):
        return 1
    else:
        return 0


def getDistance(coordYEgg1, coordYEgg2):
    dist = abs(coordYEgg1 - coordYEgg2) # Abs (produces +ve value only)
    return dist

#cap = cv2.VideoCapture('20180910_144521.mp4') # Open Video Saved in File
cap = cv2.VideoCapture(0) # Open Video Saved in File

fgbg = cv2.createBackgroundSubtractorMOG2()   # Background Subtraction in File

# Check my Tutorial
# https://github.com/SyedUmaidAhmed/Background-Subtraction-Using-OPENCV-and-Python-2.7.13/blob/master/backgroundsub.py
# Real Time Capturing using While Loop

while True:
    (grabbed, frame) = cap.read()

# This if condition will execute only when code ends, properly

    if not grabbed:
        print('Egg count: ' + str(eggCount))
        print('\n End of the video file...')
        break


    # get Settings radius/area values
    # set is our created object taken from class defined above
    # inside the class we have 'get radius, get area' regular method
    
    radius_min,radius_max = set.getRadius()
    area_min,area_max = set.getArea()

    if radius_min == '':
        radius_min = 0
    if radius_max == '':
        radius_max = 0

    if area_min == '':
        area_min = 0
    if area_max == '':
        area_max = 0

    # Calling the Function with Arguments to resize the Video Frames

    frame40 = reScaleFrame(frame, percent=40)

    #In Python, np.size() function count the number of elements along a given axis.

    height = np.size(frame40, 0)
    width = np.size(frame40, 1)

#For your Understanding , A little sample
#arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
#By default, give the total number of elements.
#print(np.size(arr))
# Answer : 8

    fgmask = fgbg.apply(frame40) # Background Subtraction
    
    hsv = cv2.cvtColor(frame40, cv2.COLOR_BGR2HSV) # Color Saturation BGR_2_HSV

# We will use V-Channel Only
    
    th, bw = cv2.threshold(hsv[:, :, 2], 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) # Apply Threshold to third list in value

# Morphological Transformations : It needs two inputs, one is our original image,
# second one is called structuring element or kernel which decides the nature of operation
# numpy can aslo generate structuring element, but for elliptical, circular and we use 'cv2.getStructuringElement'
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

# cv2.MORPH_ELLIPSE will produce a ball / circle shaped element with a diameter of 5
# cv2.MORPH_CLOSE is for closing operation in line, removing dots from 'J'
    
    morph = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)


#cv2.distanceTransform: It should be a 2D array of type np.uint8(not int8).
#cv2.distanceTransform will help in Touching egg corners
# Touching Conditions
# Stack: https://stackoverflow.com/questions/26932891/detect-touching-overlapping-circles-ellipses-with-opencv-and-python
#In this operation, the gray level intensities of the points
#inside the foreground regions are changed to distance their respective distances from the closest 0 value (boundary)

    dist = cv2.distanceTransform(morph, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

# Practical of cv2.distanceTransform - OpenCV https://www.tutorialspoint.com/opencv/opencv_distance_transformation.htm

    borderSize = set.getBorderSizeValue() # Taking it from PyQt5 Class
    
#You can add various borders to an image in using the method copyMakeBorder()
#https://www.tutorialspoint.com/opencv/opencv_adding_borders.htm
    
    distborder = cv2.copyMakeBorder(dist, borderSize, borderSize, borderSize, borderSize,
                                    cv2.BORDER_CONSTANT | cv2.BORDER_ISOLATED, 0)
    
# After making border precise we want to detect inner surface for avoidance of mixing boundaries
# Initializing a self-gap by ourselves make ellipse smaller
    gap = 10
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * (borderSize - gap) + 1, 2 * (borderSize - gap) + 1))
    kernel2 = cv2.copyMakeBorder(kernel2, gap, gap, gap, gap,
                                 cv2.BORDER_CONSTANT | cv2.BORDER_ISOLATED, 0)

    distTempl = cv2.distanceTransform(kernel2, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)


# Match Template, small circle and ellipse border

    nxcor = cv2.matchTemplate(distborder, distTempl, cv2.TM_CCOEFF_NORMED)

    mn, mx, _, _ = cv2.minMaxLoc(nxcor)
    th, peaks = cv2.threshold(nxcor, mx * 0.5, 255, cv2.THRESH_BINARY)
    peaks8u = cv2.convertScaleAbs(peaks)

    # fgmask = self.fgbg.apply(peaks8u)

# cv2.RETR_CCOMP, For extracting both internal and external contours,
# and organizing them into a two-level hierarchy

    _, contours, hierarchy = cv2.findContours(peaks8u, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

#Scales, calculates absolute values, and converts the result to 8-bit

    
    peaks8u = cv2.convertScaleAbs(peaks)  # to use as mask

    # plot reference lines (entrance and exit lines)
    # Draw Three lines, First Calculate the Center and Then do Offset +40,-40
    coordYEntranceLine = (height // 2) + OffsetRefLines
    coordYMiddleLine = (height // 2)
    coordYExitLine = (height // 2) - OffsetRefLines
    cv2.line(frame40, (0, coordYEntranceLine), (width, coordYEntranceLine), (255, 0, 0), 2)
    cv2.line(frame40, (0, coordYMiddleLine), (width, coordYMiddleLine), (0, 255, 0), 6)
    cv2.line(frame40, (0, coordYExitLine), (width, coordYExitLine), (255, 0, 0), 2)

    flag = False
    egg_list = []
    egg_index = 0

    for i in range(len(contours)):
        contour = contours[i]

        (x, y), radius = cv2.minEnclosingCircle(contour) # Taking Center & Radius
        radius = int(radius)
        #Execute it if you want understanding of above cv2.minEnclosingCircle
        # cv2.circle(frame40,(int(x),int(y)),radius,(0,255,0),2)

# Let (x,y) be the top-left coordinate of the rectangle and (w,h) be its width and height.
# For understanding of cv2.minEnclosingCircle and cv2.boundingRect
# Links: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html

        (x, y, w, h) = cv2.boundingRect(contour)

        # Execute it if you want understanding of above cv2.minEnclosingCircle
        # cv2.rectangle(frame40,(x,y),(x+w,y+h),(0,255,0),2)

        egg_index = i

        egg_list.append([x, y, flag])

        if len(contour) >= 5:

            if (radius <= int(radius_max) and radius >= int(radius_min)):

                # print("radius: ", radius)
                # pprint.pprint(hierarchy)

                ellipse = cv2.fitEllipse(contour)

# Orientation is the angle at which object is directed.
# Following method also gives the Major Axis and Minor Axis lengths.
# cv2.fitEllipse
# So is that an angle between horizontal axis and major side of rectangle(=major ellipse axis)
#is the rotation angle in fitEllipse() method.
# https://datascience.stackexchange.com/questions/85064/where-is-the-rotated-angle-actually-located-in-fitellipse-method

                (center, axis, angle) = ellipse
                
                coordXContour, coordYContour = int(center[0]), int(center[1]) #Computer the center coordinates

                # Centroid Formula Calculation

                coordXCentroid = (2 * coordXContour + w) // 2
                coordYCentroid = (2 * coordYContour + h) // 2

                
                ax1, ax2 = int(axis[0]) - 2, int(axis[1]) - 2


                
                orientation = int(angle)
                area = cv2.contourArea(contour)

                if area >= int(area_min) and area <= int(area_max):
                    
                    #print('egg list: ' + str(egg_list) + ' index: ' + str(egg_index))

                    if CheckInTheArea(coordYContour, coordYEntranceLine, coordYExitLine): # Checking Done Always Inside the Boundary
                        
                        cv2.ellipse(frame40, (coordXContour, coordYContour), (ax1, ax2), orientation, 0, 360,
                                    (255, 0, 0), 2)  # blue around the real egg
                        
                        cv2.circle(frame40, (coordXContour, coordYContour), 1, (0, 255, 0), 15)  # green Center Marked
                        
                        cv2.putText(frame40, str(int(area)), (coordXContour, coordYContour), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, 0, 1, cv2.LINE_AA)

                    for k in range(len(egg_list)): #Checking the array in which x,y and flag are putted in past
                        egg_new_X = x
                        egg_new_Y = y

                        dist = getDistance(egg_new_Y, egg_list[k][1]) #This function returns the +ve distance between two

                        if dist > distance_tresh:  # distance_tresh = 200
                            egg_list.append([egg_new_X, egg_new_Y, flag])

                    if CheckEntranceLineCrossing(egg_list[egg_index][1], coordYMiddleLine) and not egg_list[egg_index][
                        2]:
                        eggCount += 1
                        egg_list[egg_index][2] = True

                cv2.putText(frame40, "Eggs Counted: {}".format(str(eggCount)), (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (250, 0, 255), 2)

    cv2.imshow("Original Frame", frame40)
    #cv2.imshow("Background Mask", fgmask)
    #cv2.imshow("HSV", hsv)
    #cv2.imshow("Thresh",bw)
    cv2.imshow("Th",peaks)
    cv2.imshow("copymakeBorder",distborder)
    cv2.imshow("Distance_trans",distTempl)

    key = cv2.waitKey(1)

    if key == 27:
        break

# cleanup the camera and close any open windows
cap.release()
cv2.destroyAllWindows()




