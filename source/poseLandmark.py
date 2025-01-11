import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.v2 as v2

import cv2
from PIL import Image

class poseLandmark(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = transforms.Compose([
            v2.PILToTensor(),
            v2.ToDtype(torch.float32),
            v2.Resize(256)
        ])

        self.centralFeatureOut = 1024
        self.centralCoordinateOut = 1024
        self.hiddenNodes = 32

        self.centralFeature = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(64),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(256),

            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(384),

            nn.Conv2d(in_channels=384, out_channels=512, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(512),

            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.ReLU(),
            nn.Conv2d(in_channels=1024, out_channels=self.centralFeatureOut, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.AdaptiveMaxPool2d(8),
            nn.Flatten()
        )
        self.centralCoordinate = nn.Sequential(
            nn.Linear(8*8*self.centralFeatureOut, 1024),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.Linear(1024, self.centralCoordinateOut),
            nn.Dropout(0.4),
            nn.LeakyReLU(),
        )

        self.headCoordinate = nn.Sequential(
            nn.Linear(self.centralCoordinateOut, self.hiddenNodes),
            nn.Dropout(0.4),
            nn.Tanh(),
            nn.Linear(self.hiddenNodes, 2),
            nn.Sigmoid()
        )
        self.neckCoordinate = nn.Sequential(
            nn.Linear(self.centralCoordinateOut, self.hiddenNodes),
            nn.Dropout(0.4),
            nn.Tanh(),
            nn.Linear(self.hiddenNodes, 2),
            nn.Sigmoid()
        )
        self.torsoCoordinate = nn.Sequential(
            nn.Linear(self.centralCoordinateOut, self.hiddenNodes),
            nn.Dropout(0.4),
            nn.Tanh(),
            nn.Linear(self.hiddenNodes, 2),
            nn.Sigmoid()
        )

        self.leftShoulderCoordinate = nn.Sequential(
            nn.Linear(self.centralCoordinateOut, self.hiddenNodes),
            nn.Dropout(0.4),
            nn.Tanh(),
            nn.Linear(self.hiddenNodes, 2),
            nn.Sigmoid()
        )
        self.leftElbowCoordinate = nn.Sequential(
            nn.Linear(self.centralCoordinateOut, self.hiddenNodes),
            nn.Dropout(0.4),
            nn.Tanh(),
            nn.Linear(self.hiddenNodes, 2),
            nn.Sigmoid()
        )
        self.leftWristCoordinate = nn.Sequential(
            nn.Linear(self.centralCoordinateOut, self.hiddenNodes),
            nn.Dropout(0.4),
            nn.Tanh(),
            nn.Linear(self.hiddenNodes, 2),
            nn.Sigmoid()
        )

        self.rightShoulderCoordinate = nn.Sequential(
            nn.Linear(self.centralCoordinateOut, self.hiddenNodes),
            nn.Dropout(0.4),
            nn.Tanh(),
            nn.Linear(self.hiddenNodes, 2),
            nn.Sigmoid()
        )
        self.rightElbowCoordinate = nn.Sequential(
            nn.Linear(self.centralCoordinateOut, self.hiddenNodes),
            nn.Dropout(0.4),
            nn.Tanh(),
            nn.Linear(self.hiddenNodes, 2),
            nn.Sigmoid()
        )
        self.rightWristCoordinate = nn.Sequential(
            nn.Linear(self.centralCoordinateOut, self.hiddenNodes),
            nn.Dropout(0.4),
            nn.Tanh(),
            nn.Linear(self.hiddenNodes, 2),
            nn.Sigmoid()
        )

        self.leftHipCoordinate = nn.Sequential(
            nn.Linear(self.centralCoordinateOut, self.hiddenNodes),
            nn.Dropout(0.4),
            nn.Tanh(),
            nn.Linear(self.hiddenNodes, 2),
            nn.Sigmoid()
        )
        self.leftKneeCoordinate = nn.Sequential(
            nn.Linear(self.centralCoordinateOut, self.hiddenNodes),
            nn.Dropout(0.4),
            nn.Tanh(),
            nn.Linear(self.hiddenNodes, 2),
            nn.Sigmoid()
        )
        self.leftHeelCoordinate = nn.Sequential(
            nn.Linear(self.centralCoordinateOut, self.hiddenNodes),
            nn.Dropout(0.4),
            nn.Tanh(),
            nn.Linear(self.hiddenNodes, 2),
            nn.Sigmoid()
        )
        self.leftFootToeCoordinate = nn.Sequential(
            nn.Linear(self.centralCoordinateOut, self.hiddenNodes),
            nn.Dropout(0.4),
            nn.Tanh(),
            nn.Linear(self.hiddenNodes, 2),
            nn.Sigmoid()
        )

        self.rightHipCoordinate = nn.Sequential(
            nn.Linear(self.centralCoordinateOut, self.hiddenNodes),
            nn.Dropout(0.4),
            nn.Tanh(),
            nn.Linear(self.hiddenNodes, 2),
            nn.Sigmoid()
        )
        self.rightKneeCoordinate = nn.Sequential(
            nn.Linear(self.centralCoordinateOut, self.hiddenNodes),
            nn.Dropout(0.4),
            nn.Tanh(),
            nn.Linear(self.hiddenNodes, 2),
            nn.Sigmoid()
        )
        self.rightHeelCoordinate = nn.Sequential(
            nn.Linear(self.centralCoordinateOut, self.hiddenNodes),
            nn.Dropout(0.4),
            nn.Tanh(),
            nn.Linear(self.hiddenNodes, 2),
            nn.Sigmoid()
        )
        self.rightFootToeCoordinate = nn.Sequential(
            nn.Linear(self.centralCoordinateOut, self.hiddenNodes),
            nn.Dropout(0.4),
            nn.Tanh(),
            nn.Linear(self.hiddenNodes, 2),
            nn.Sigmoid()
        )

    def preprocessImage(self, imageMat):
        colorConverted = cv2.cvtColor(imageMat, cv2.COLOR_BGR2RGB)
        transformedImage = self.transformer(Image.fromarray(colorConverted).convert("RGB")) / 255
        return transformedImage

    def forward(self, x):
        x = self.centralFeature(x)
        x = self.centralCoordinate(x)

        headCoordinate = self.headCoordinate(x)
        neckCoordinate = self.neckCoordinate(x)
        torsoCoordinate = self.torsoCoordinate(x)
        leftShoulderCoordinate = self.leftShoulderCoordinate(x)
        leftElbowCoordinate = self.leftElbowCoordinate(x)
        leftWristCoordinate = self.leftWristCoordinate(x)
        rightShoulderCoordinate = self.rightShoulderCoordinate(x)
        rightElbowCoordinate = self.rightElbowCoordinate(x)
        rightWristCoordinate = self.rightWristCoordinate(x)
        leftHipCoordinate = self.leftHipCoordinate(x)
        leftKneeCoordinate = self.leftKneeCoordinate(x)
        leftHeelCoordinate = self.leftHeelCoordinate(x)
        leftFootToeCoordinate = self.leftFootToeCoordinate(x)
        rightHipCoordinate = self.rightHipCoordinate(x)
        rightKneeCoordinate = self.rightKneeCoordinate(x)
        rightHeelCoordinate = self.rightHeelCoordinate(x)
        rightFootToeCoordinate = self.rightFootToeCoordinate(x)

        return headCoordinate, neckCoordinate, torsoCoordinate,leftShoulderCoordinate, leftElbowCoordinate, leftWristCoordinate,rightShoulderCoordinate, rightElbowCoordinate, rightWristCoordinate,leftHipCoordinate, leftKneeCoordinate, leftHeelCoordinate, leftFootToeCoordinate,rightHipCoordinate, rightKneeCoordinate, rightHeelCoordinate, rightFootToeCoordinate