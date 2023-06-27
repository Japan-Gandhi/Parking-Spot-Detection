import cv2 as cv
import numpy as np
import pandas as pd
import pickle
import os

# Saving Height and width
height = 48
width = 108


with open("Parking-Spot-Detection/parkingSpotList.p", "rb") as file:
    positionList = pickle.load(file)

cap = cv.VideoCapture(
    "Parking-Spot-Detection/Resources/Car Park Stablized.mp4")


def checkSpotAvailability(imgThreshhold):
    for pos in positionList:
        xCord, yCord = pos
        threshCrop = imgThreshhold[yCord:yCord+height+1, xCord: xCord+width+1]
        nonZeroCount = cv.countNonZero(threshCrop)
        cv.putText(frame, str(nonZeroCount), (xCord+10, yCord+40),
                   cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 255), 1)

        if nonZeroCount > 300:
            cv.rectangle(frame, pos, (xCord+width, yCord+height), (0, 0, 255), 2)
        else:
            cv.rectangle(frame, pos, (xCord+width, yCord+height), (0,255,0), 2)



while True:

    ret, frame = cap.read()
    if cap.get(cv.CAP_PROP_POS_FRAMES,) == cap.get(cv.CAP_PROP_FRAME_COUNT):
        cap.set(cv.CAP_PROP_POS_FRAMES, 0)

    k = 3
    imgGray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    imgBlur = cv.blur(imgGray, (k, k), 1)
    imgThreshhold = cv.adaptiveThreshold(
        imgBlur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 25, 24)

    checkSpotAvailability(imgThreshhold)

    cv.imshow("Video Capture", frame)
    if cv.waitKey(10) == 13:  # Enter Key
        break


cv.destroyAllWindows()
