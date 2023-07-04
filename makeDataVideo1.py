import cv2 as cv
import numpy as np
import pandas as pd
import pickle
import os

# Saving Height and width
height = 47
width = 90

# Parking Spot images naming index

emptySpotCount = 1
occupiedSpotCount = 1
skipCounter = 1

# Directory Path for dataset
dirPath = "C:\D\College Stuff\Semester 4\Summer Internship\Parking-Spot-Detection\Resources\dataset3"


with open("Parking-Spot-Detection/parkingSpotListVideo1.p", "rb") as file:
    positionList = pickle.load(file)

cap = cv.VideoCapture(
    "Parking-Spot-Detection/Resources/Video 1 - Walmart Parking Lot.mp4")


def checkSpotAvailability(imgThreshhold):

    # Re-initiating the variables
    global occupiedSpotCount, emptySpotCount, skipCounter

    for pos in positionList:
        xCord, yCord = pos
        threshCrop = imgThreshhold[yCord:yCord+height+1, xCord: xCord+width+1]
        nonZeroCount = cv.countNonZero(threshCrop)
        # cv.putText(frame, str(nonZeroCount), (xCord+10, yCord+40),
        #            cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 255), 1)

        if nonZeroCount > 150:

            if skipCounter % 40 == 0:
                # Declaring the path name to the required directory
                occupiedPath = os.path.join(dirPath, "occupied")

                # Naming the file
                occFileName = str(occupiedSpotCount) + ".jpg"

                # Changing the Directory
                os.chdir(occupiedPath)

                # Saving the file
                cv.imwrite(
                    occFileName, frame[yCord:yCord + height + 1, xCord: xCord + width + 1])
                occupiedSpotCount += 1
                # cv.rectangle(frame, pos, (xCord+width, yCord+height), (0, 0, 255), 2)
            skipCounter += 1

        else:
            # Declaring the path name to the required directory
            emptyPath = os.path.join(dirPath, "empty")

            # Naming the file
            empFileName = str(emptySpotCount) + ".jpg"

            # Changing the Directory
            os.chdir(emptyPath)

            # Saving the file
            cv.imwrite(
                empFileName, frame[yCord:yCord + height + 1, xCord: xCord + width + 1])
            emptySpotCount += 1
            # cv.rectangle(frame, pos, (xCord+width, yCord+height), (0,255,0), 2)


while True:

    ret, frame = cap.read()
    # if cap.get(cv.CAP_PROP_POS_FRAMES,) == cap.get(cv.CAP_PROP_FRAME_COUNT):
    #     cap.set(cv.CAP_PROP_POS_FRAMES, 0)

    imgGray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    imgBlur = cv.blur(imgGray, (3, 3), 1)
    imgThreshhold = cv.adaptiveThreshold(
        imgBlur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 25, 24)

    checkSpotAvailability(imgThreshhold)

    cv.imshow("Video Capture", frame)
    if cv.waitKey(100) == 13:  # Enter Key
        break


cv.destroyAllWindows()
