import torch
import torch.nn as nn

class EmotionCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(32,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(64,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(128,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.fc_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256,512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512,7)
        )

    def forward(self,x):

        x=self.block1(x)
        x=self.block2(x)
        x=self.block3(x)
        x=self.block4(x)
        x=self.global_pool(x)
        x=self.fc_block(x)

        return x
