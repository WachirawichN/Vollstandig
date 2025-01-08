import torch
import torchvision.transforms as transforms
import torchvision.transforms.v2 as v2
from poseLandmark import poseLandmark

import cv2
from PIL import Image

from pathlib import Path


def predictLandmarks(model, transformedImage, device):
    transformedImage = transformedImage.unsqueeze(0) # Simulate being a batch
    transformedImage = transformedImage.to(device)
    model = model.to(device)
    model.eval()
    landmarks = model(transformedImage)
    return landmarks
def connectPoseLandmarks(imageMat, pixelLandmarks, lineThickness):
    # Connection diagram:
    # Head -> Neck -> Torso -> Left shoulder    -> Left elbow   -> Left wrist
    #                       -> Right shoulder   -> Right elbow  -> Right wrist
    #                       -> Left hip         -> Left Knee    -> Left heel    -> Left foot toe tip
    #                       -> Right hip        -> Right Knee   -> Right heel   -> Right foot toe tip
    for index in range(2):
        imageMat = cv2.line(imageMat, pixelLandmarks[index], pixelLandmarks[index + 1], (0, 255, 0), lineThickness)

    for index in range(3, 5, 1):
        imageMat = cv2.line(imageMat, pixelLandmarks[index], pixelLandmarks[index + 1], (255, 0, 0), lineThickness)
    for index in range(6, 8, 1):
        imageMat = cv2.line(imageMat, pixelLandmarks[index], pixelLandmarks[index + 1], (0, 0, 255), lineThickness)

    for index in range(9, 12, 1):
        imageMat = cv2.line(imageMat, pixelLandmarks[index], pixelLandmarks[index + 1], (255, 0, 0), lineThickness)
    for index in range(13, 16, 1):
        imageMat = cv2.line(imageMat, pixelLandmarks[index], pixelLandmarks[index + 1], (0, 0, 255), lineThickness)

    imageMat = cv2.line(imageMat, pixelLandmarks[2], pixelLandmarks[3], (255, 0, 0), lineThickness)
    imageMat = cv2.line(imageMat, pixelLandmarks[2], pixelLandmarks[6], (0, 0, 255), lineThickness)
    imageMat = cv2.line(imageMat, pixelLandmarks[2], pixelLandmarks[9], (255, 0, 0), lineThickness)
    imageMat = cv2.line(imageMat, pixelLandmarks[2], pixelLandmarks[13], (0, 0, 255), lineThickness)
    return imageMat

def readVideo(model, device, videoPATH, transformer):
    model.to(device)

    videoCap = cv2.VideoCapture(videoPATH)
    fps = videoCap.get(cv2.CAP_PROP_FPS)
    windowName = "video"
    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)

    success, imageMat = videoCap.read()
    while success:
        width, height = imageMat.shape[1], imageMat.shape[0]
        colorConverted = cv2.cvtColor(imageMat, cv2.COLOR_BGR2RGB)
        transformedImage = transformer(Image.fromarray(colorConverted).convert("RGB")) / 255

        landmarks = predictLandmarks(model, transformedImage, device)
        pixelLandmarks = []
        for landmark in landmarks:
            xCoord = int(landmark[0][0].item() * width)
            yCoord = int(landmark[0][1].item() * height)
            pixelLandmarks.append((xCoord, yCoord))

        # Draw dot on each landmark
        for landmark in pixelLandmarks:
            imageMat = cv2.circle(imageMat, landmark, 20, (0, 255, 0), -1)
        # Connect those dot
        imageMat = connectPoseLandmarks(imageMat, pixelLandmarks, 15)

        cv2.imshow(windowName, imageMat)
        k = cv2.waitKey(1)
        if k == 113:
            break
        success, imageMat = videoCap.read()

    videoCap.release()
    cv2.destroyAllWindows()


parentDir = Path(__file__).parent
videoDir = parentDir.joinpath("dataset-generator/data/video-converted/20240928_105700.mp4")

model = poseLandmark()
model.load_state_dict(torch.load(parentDir.joinpath("model/poseLandmarks-Temp.pth")))
transformer = transforms.Compose([
    v2.PILToTensor(),
    v2.ToDtype(torch.float32),
    v2.Resize(256)
])
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
    
readVideo(model, device, videoDir, transformer)