import tkinter as tk
from tkinter import *
from tkinter import ttk
from PIL import ImageTk, Image

import cv2 as cv
import numpy as np
import pandas as pd
import pickle
import os
import tensorflow as tf
import keras
from keras.models import load_model
import time


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

ret, frame = cap.read()
flag = 0


def checkSpotAvailability(imgFrame):

    progress = ttk.Progressbar(
        root, orient="horizontal", mode="determinate", length=150)
    progress.grid(row=1, column=1, pady=10)
    

    global ret, frame

    spotCounter = 0
    emptyCounter = 0
    global flag

    flag = 1

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

        progress["value"] += float((100/12))
        root.update_idletasks()
        

        if spotCounter == 12:
            break

    cv.putText(frame, "Free Spots: {}/{}".format(emptyCounter, spotCounter),
               (30, 30), cv.FONT_HERSHEY_COMPLEX, 1, (45, 25, 255), 2)

    # resized_frame = cv.resize(imgFrame, (widthW, heightW))
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    img = Image.fromarray(frame_rgb)
    img_tk = ImageTk.PhotoImage(image=img)

    video_label.configure(image=img_tk)
    video_label.grid(column=0, row=2, columnspan=2)
    
    
    video_label.image = img_tk

    flag = 0
    root.update_idletasks()
    time.sleep(3)

    update_frame()

    # cv.waitKey(5000)


def predictStatus(image):

    modelLocation = "C:\D\College Stuff\Semester 4\Summer Internship\Parking-Spot-Detection\models\parkingSpotClassifier.h5"
    model = load_model(modelLocation)

    # print(image.shape)

    imageResize = tf.image.resize(image, (49, 109))
    prediction = model.predict(np.expand_dims(imageResize/255, 0))
    predictionValue = prediction[0][0]

    return predictionValue


def update_frame():

    global ret, frame
    global flag
    
    if cap.get(cv.CAP_PROP_POS_FRAMES,) == cap.get(cv.CAP_PROP_FRAME_COUNT):
        cap.set(cv.CAP_PROP_POS_FRAMES, 0)

    if flag == 0:

        ret, frame = cap.read()
        if ret:
            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            resized_frame = cv.resize(frame_rgb, (widthW, heightW))

            img = Image.fromarray(resized_frame)
            img_tk = ImageTk.PhotoImage(image=img)

            video_label.configure(image=img_tk)
            video_label.grid(column=0, row=2, columnspan=2)
            video_label.image = img_tk

            root.after(30, update_frame)


clrBeige = "#f5f5dc"


# Beginning of the GUI
root = tk.Tk()
root.title("Gandhi-Nagpal Virtual Valet Pro")
# root.geometry("1400x800")
root.config(bg=clrBeige)

# Heading
heading = Label(root, text="Gandhi-Nagpal Virtual Valet")
heading.grid(row=0, column=0, padx=20, pady=10, columnspan=2)
heading.config(font=("Lexend Deca", 30), bg=clrBeige)


widthW = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
heightW = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

video_label = tk.Label(root)
video_label.grid(row=2, column=0, pady=10)

# Enter Car Button
btn = Button(root, text="Car Enter",
             command=lambda: checkSpotAvailability(frame))
btn.grid(column=0, row=1, padx=20, pady=10)
btn.config(font=("Lexend Deca", 15))


update_frame()


root.mainloop()
