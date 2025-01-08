# This script only support mp4 file format, for now

import cv2
from pathlib import Path
from tqdm import tqdm
from glob import glob

def getFiles(targetDir):
    return glob(f"{targetDir}/*.mp4")

dataDir = Path(__file__).parent.joinpath("data")
videoDir = dataDir.joinpath("video")
imageDir = dataDir.joinpath("image")
resizeRatio = 4

video = getFiles(videoDir)
videoTqdm = tqdm(video, desc="Processing video", unit="video")
for video in videoTqdm:
    videoName = Path(video).name.split(".")[0]
    videoCap = cv2.VideoCapture(video)
    width  = int(videoCap.get(cv2.CAP_PROP_FRAME_WIDTH) / resizeRatio)
    height = int(videoCap.get(cv2.CAP_PROP_FRAME_HEIGHT) / resizeRatio)

    success, image = videoCap.read()
    frameNo = 0
    while success:
        image = cv2.resize(image, (width , height))
        fileName = str(imageDir.joinpath(f"{videoName}_{frameNo}.jpg"))
        cv2.imwrite(fileName, image)
        success, image = videoCap.read()
        frameNo += 1