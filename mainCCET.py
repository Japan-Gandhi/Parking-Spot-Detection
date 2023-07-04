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
height = 130
width = 289


# Loading the spot list from the pickle file
with open("Parking-Spot-Detection/parkingSpotListCCET.p", "rb") as file:
    positionList = pickle.load(file)

# Importing the input video file
cap = cv.VideoCapture(
    "Parking-Spot-Detection/Resources/02.mp4")

ret, frame = cap.read()
flag = 0


def checkSpotAvailability(imgFrame):

    # videoDisplay.grid(row=3, column=0, pady=10, columnspan=2)
    # progress.configure(troughcolor='#E0E0E0', background='#00BFFF', troughrelief='flat', relief='flat')
    
    global ret, frame

    spotCounter = 0
    emptyCounter = 0
    global flag
    
    posListLength = len(positionList)

    flag = 1

    for pos in positionList:
        spotCounter += 1
        xCord, yCord = pos

        singleSpot = imgFrame[yCord:yCord+height+1, xCord: xCord+width+1]
        prediction = predictStatus(singleSpot)
        print(spotCounter, prediction)

        if prediction > 0.5:
            cv.rectangle(frame, pos, (xCord+width,
                         yCord+height), (0, 0, 255), 2)
            cv.putText(frame, str("Occupied"), (xCord+10, yCord+40),
                       cv.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 0), 1)
            pass

        else:
            cv.rectangle(frame, pos, (xCord+width,
                         yCord+height), (0, 255, 0), 2)
            spotNumber = str(spotCounter)
            cv.putText(frame, spotNumber, (xCord+40, yCord+30),
                       cv.FONT_HERSHEY_COMPLEX, 0.7, (0,255,255), 1)

            emptyCounter += 1

        progress["value"] += float((100/posListLength))
        root.update_idletasks()

        # if spotCounter == 12:
        #     break

    cv.putText(frame, "Free Spots: {}/{}".format(emptyCounter, spotCounter),
               (30, 30), cv.FONT_HERSHEY_COMPLEX, 1, (45, 25, 255), 2)

    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    img_tk = ImageTk.PhotoImage(image=img)

    videoDisplay.configure(image=img_tk)
    videoDisplay.grid(row=3, column=0, pady=10, columnspan=2)
    videoDisplay.image = img_tk

    flag = 0
    root.update_idletasks()
    time.sleep(10)
    progress["value"] =0
    updateFrame()


def predictStatus(image):

    modelLocation = "Parking-Spot-Detection\models\parkingSpotClassifierV2.h5"
    model = load_model(modelLocation)

    # print(image.shape)

    imageResize = tf.image.resize(image, (49, 109))
    prediction = model(np.expand_dims(imageResize/255, 0))
    predictionValue = prediction[0][0]

    return predictionValue


def updateFrame():

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

            videoDisplay.configure(image=img_tk)
            videoDisplay.grid(row=3, column=0, pady=10, columnspan=2)
            videoDisplay.image = img_tk

            root.after(100, updateFrame)


clrBeige = "#f5f5dc"


# Beginning of the GUI
root = tk.Tk()
root.title("Gandhi-Nagpal Virtual Valet Pro")
root.config(bg=clrBeige)


# screen_width = root.winfo_screenwidth()
# screen_height = root.winfo_screenheight()
# root.geometry(f"{screen_width}x{screen_height}")

# Heading
heading = Label(root, text="GANDHI-NAGPAL VIRTUAL VALET")
heading.grid(row=0, column=0, padx=20, columnspan=2)
heading.config(font=("Lexend Deca", 30), bg=clrBeige, anchor="center")


# Enter Car Button
btn = Button(root, text="Car Enter",
             command=lambda: checkSpotAvailability(frame))
btn.grid(column=0, row=1, padx=20, columnspan=2)
btn.config(font=("Lexend Deca", 15))

progress = ttk.Progressbar(
    root, orient="horizontal", mode="determinate", length=250)
progress.grid(row=2, column=0, pady=10, columnspan=2)



# Video Display
widthW = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
heightW = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

videoDisplay = tk.Label(root)
# videoDisplay.grid(row=3, column=0, pady=10)




updateFrame()
root.mainloop()
