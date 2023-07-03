import cv2 as cv
import numpy as np
import pandas as pd
import pickle
import os
import tensorflow as tf
import keras
from keras.models import load_model

# Saving Height and width
height = 48
width = 108

# Loading the spot list from the pickle file
with open("Parking-Spot-Detection/parkingSpotList.p", "rb") as file:
    positionList = pickle.load(file)

positionList.reverse()
# Importing the input video file
cap = cv.VideoCapture(
    "Parking-Spot-Detection/Resources/Car Park Stablized.mp4")


def checkSpotAvailability(imgFrame):

    spotCounter = 0
    emptyCounter = 0

    for pos in positionList:
        spotCounter += 1
        xCord, yCord = pos

        singleSpot = imgFrame[yCord:yCord+height+1, xCord: xCord+width+1]
        prediction = predictStatus(singleSpot)

        if prediction > 0.5:
            cv.rectangle(frame, pos, (xCord+width,
                         yCord+height), (0, 0, 255), 2)
            cv.putText(frame, str("Occupied"), (xCord+10, yCord+40),
                       cv.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 0), 1)

        else:
            cv.rectangle(frame, pos, (xCord+width,
                         yCord+height), (0, 255, 0), 2)
            cv.putText(frame, str("Empty"), (xCord+10, yCord+40),
                       cv.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 0), 1)
            emptyCounter += 1
        
    cv.putText(frame, "Free Spots: {}/{}".format(emptyCounter, spotCounter), (30, 30),
               cv.FONT_HERSHEY_COMPLEX, 1, (45, 25, 255), 2)
    


def predictStatus(image):

    modelLocation = "C:\D\College Stuff\Semester 4\Summer Internship\Parking-Spot-Detection\models\parkingSpotClassifier.h5"
    model = load_model(modelLocation)
    
    # print(image.shape)

    imageResize = tf.image.resize(image, (49, 109))
    prediction = model.predict(np.expand_dims(imageResize/255, 0))
    predictionValue = prediction[0][0]
    
    return predictionValue


# Driver Loop
while True:
    ret, frame = cap.read()
    if cap.get(cv.CAP_PROP_POS_FRAMES,) == cap.get(cv.CAP_PROP_FRAME_COUNT):
        cap.set(cv.CAP_PROP_POS_FRAMES, 0)

    checkSpotAvailability(frame)

    cv.imshow("Video Capture", frame)
    if cv.waitKey(10) == 13:  # Enter Key
        break


# End (close all windows)
cv.destroyAllWindows()


