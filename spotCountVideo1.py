import cv2 as cv
import numpy as np
import pandas as pd
import pickle

# Saving Height and width
height = 47
width = 90


# Checking the existance of list of parking spots in a pickle file.
try:
    with open("Parking-Spot-Detection/parkingSpotListVideo1.p", "rb") as file:
        positionList = pickle.load(file)

except:
    positionList = []


def mouseClickCallback(event, x, y, flags, param):

    if event == cv.EVENT_RBUTTONDOWN:
        currentPos = (x, y)
        positionList.append(currentPos)

    if event == cv.EVENT_LBUTTONDOWN:

        index = 0
        for pos in positionList:

            x1, y1 = pos
            if (x1 <= x <= x1+width) and (y1 <= y <= y1 + height):
                positionList.pop(index)
            index += 1


while True:

    image = cv.imread("Parking-Spot-Detection\Resources\input_image_V1.png")

    # Manual Calculation of Height and Width
    # cv.rectangle(image, (50, 142), (158, 192), (0, 0, 255), 1)

    for position in positionList:
        cv.rectangle(image, position,
                     (position[0]+width, position[1]+height), (0, 255, 0), 2)

    cv.imshow("Input Image", image)
    cv.setMouseCallback("Input Image", mouseClickCallback)

    if cv.waitKey(10) == 13:  # Enter Key
        break


with open("Parking-Spot-Detection/parkingSpotListVideo1.p", "wb") as file:
    pickle.dump(positionList, file)


cv.destroyAllWindows()
