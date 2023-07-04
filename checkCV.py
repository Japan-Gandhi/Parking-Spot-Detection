import cv2 as cv
import numpy as np
import pandas as pd
import pickle
import os


# Saving Height and width
height = 30
width = 50
# Loading the spot list from the pickle file
with open("Parking-Spot-Detection/parkingSpotListVideo2.p", "rb") as file:
    positionList = pickle.load(file)

# Importing the input video file
cap = cv.VideoCapture(
    "Parking-Spot-Detection/Resources/Video 2 - Busy Parking Lot.mp4")

def checkSpotAvailability(imgThreshhold):
    
    spotCounter = 0
    emptyCounter = 0
    for pos in positionList:
        spotCounter += 1
        xCord, yCord = pos
        threshCrop = imgThreshhold[yCord:yCord+height+1, xCord: xCord+width+1]
        nonZeroCount = cv.countNonZero(threshCrop)
        cv.putText(frame, str(nonZeroCount), (xCord+10, yCord+40),
                   cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 255), 1)

        if nonZeroCount > 170:
            cv.rectangle(frame, pos, (xCord+width, yCord+height), (0, 0, 255), 1)
           

            
        else:
            cv.rectangle(frame, pos, (xCord+width, yCord+height), (0,255,0), 1)
            emptyCounter += 1

    cv.putText(frame, "Free Spots: {}/{}".format(emptyCounter,spotCounter), (30,30),
                   cv.FONT_HERSHEY_COMPLEX, 0.8, (45, 25, 255), 1)


while True:
    ret, frame = cap.read()
    if cap.get(cv.CAP_PROP_POS_FRAMES,) == cap.get(cv.CAP_PROP_FRAME_COUNT):
        cap.set(cv.CAP_PROP_POS_FRAMES, 0)

    imgGray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    imgBlur = cv.blur(imgGray, (3,3), 1)
    imgThreshhold = cv.adaptiveThreshold(
        imgBlur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 25, 24)

    checkSpotAvailability(imgThreshhold)

    cv.imshow("Video Capture", frame)
    # cv.imshow("Video Capture", imgThreshhold)
    if cv.waitKey(100) == 13:  # Enter Key
        break

cv.destroyAllWindows()
