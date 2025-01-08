# Left mouse = Select position for the landmark coordinate.
# Middle mouse = Undo.
# Right mouse = Skip current landmark.
# Any key = Next image.
# Q = Quit.

from pathlib import Path
from glob import glob
import os
import pandas as pd
import cv2
import random

def addAxis(label):
    newDict = {}
    for axis in ["x", "y"]:
        newDict[f"{label}_{axis}"] = []
    return newDict

def getFiles(targetDir):
    return glob(f"{targetDir}/*.jpg")

# Landmark selection
poseLandmarkLabels = ["head",
                      "neck",
                      "torso",
                      "left_shoulder",
                      "left_elbow",
                      "left_wrist",
                      "right_shoulder",
                      "right_elbow",
                      "right_wrist",
                      "left_hip",
                      "left_knee",
                      "left_heel",
                      "left_foot_toe",
                      "right_hip",
                      "right_knee",
                      "right_heel",
                      "right_foot_toe"]
"""
poseLandmarkLabel = {11 : "left_shoulder",
                     12 : "right_shoulder",
                     13 : "left_elbow",
                     14 : "right_elbow",}
handLandmarkLabel = {0 : "wrist",
                     1 : "thumb_cmc",
                     2 : "thumb_mcp",
                     3 : "thumb_ip",
                     4 : "thumb_tip",
                     5 : "index_finger_mcp",
                     6 : "index_finger_pip",
                     7 : "index_finger_dip",
                     8 : "index_finger_tip",
                     9 : "middle_finger_mcp",
                     10 : "middle_finger_pip",
                     11 : "middle_finger_dip",
                     12 : "middle_finger_tip",
                     13 : "ring_finger_mcp",
                     14 : "ring_finger_pip",
                     15 : "ring_finger_dip",
                     16 : "ring_finger_tip",
                     17 : "pinky_mcp",
                     18 : "pinky_pip",
                     19 : "pinky_dip",
                     20 : "pinky_tip"}
"""

# Data dictionary preparation for turing into csv later
dataDict = {"image" : []}
for landmark in poseLandmarkLabels:
    dataDict = dataDict | addAxis(landmark)
"""
for key in poseLandmarkLabel:
    dataDict = dataDict | addAxis(poseLandmarkLabel[key])
for side in ["left", "right"]:
    for key in handLandmarkLabel:
        dataDict = dataDict | addAxis(f"{side}_{handLandmarkLabel[key]}")
"""
totalCooridinate =  len(dataDict)

# Files and directories
dataDir = Path(__file__).parent.joinpath("data")
outputDir = dataDir.joinpath("poseLandmarkDataset")
imageDir = dataDir.joinpath("image")
images = getFiles(imageDir)
random.shuffle(images)
merging = False
for file in os.scandir(outputDir):
    if file.is_file() and file.name == "coordinate.csv":
        merging = True

# Image processing
width, height = 0, 0
currentLandmarkIndex = 0
maxLandmarkIndex = (totalCooridinate - 1) / 2
windowName = "image"
cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)

def eventHandler(event, x, y, flags, param):
    global width, height, currentLandmarkIndex, maxLandmarkIndex, dataDict

    if event in [cv2.EVENT_LBUTTONDOWN, cv2.EVENT_MBUTTONDOWN, cv2.EVENT_RBUTTONDOWN]:
        if currentLandmarkIndex < maxLandmarkIndex:
            if event == cv2.EVENT_LBUTTONDOWN: # Add new coordinate
                xPos, yPos = x / width, y/ height
                dataDict[param[currentLandmarkIndex * 2 + 1]].append(xPos)
                dataDict[param[currentLandmarkIndex * 2 + 2]].append(yPos)
                currentLandmarkIndex += 1

            elif event == cv2.EVENT_RBUTTONDOWN: # Skip this coordinate
                dataDict[param[currentLandmarkIndex * 2 + 1]].append(0)
                dataDict[param[currentLandmarkIndex * 2 + 2]].append(0)
                currentLandmarkIndex += 1

        if event == cv2.EVENT_MBUTTONDOWN: # Undo last coordinate
            if currentLandmarkIndex > 0:
                currentLandmarkIndex -= 1
                dataDict[param[currentLandmarkIndex * 2 + 1]].pop(-1)
                dataDict[param[currentLandmarkIndex * 2 + 2]].pop(-1)

        if currentLandmarkIndex >= maxLandmarkIndex:
            print("Got all landmark next frame please")
        else:
            print(f"Total landmark: {currentLandmarkIndex}/{maxLandmarkIndex}, Current target landmark: {list(dataDict.keys())[currentLandmarkIndex * 2 + 1][:-2]}")

cv2.setMouseCallback(windowName, eventHandler, list(dataDict.keys()))
for index, imageFile in enumerate(images):
    image = cv2.imread(imageFile)
    width, height = image.shape[1], image.shape[0]
    cv2.imshow(windowName, image)
    print(f"Image {index}/{len(images)}, Name: {Path(imageFile).name}")
    print(f"Total landmark: {currentLandmarkIndex}/{maxLandmarkIndex}, Current target landmark: {list(dataDict.keys())[currentLandmarkIndex * 2 + 1][:-2]}")
    
    k = cv2.waitKey(0)
    if k == 113: # Ascii for Q
        cv2.destroyAllWindows()
        break
    else:
        currentLandmarkIndex = 0
        imagePath = Path(imageFile)
        dataDict["image"].append(imagePath.name)
        imagePath.rename(outputDir.joinpath(imagePath.name)) # Move image to output dir

df = pd.DataFrame(dataDict)
if merging:
    readDf = pd.read_csv(outputDir.joinpath("coordinate.csv"))
    newDf = pd.concat([readDf, df], axis=0)
    #newDf = newDf.sample(frac=1) # Shuffle dataset
    newDf.to_csv(outputDir.joinpath("coordinate.csv"), index=False)
else:
    df.to_csv(outputDir.joinpath("coordinate.csv"), index=False)