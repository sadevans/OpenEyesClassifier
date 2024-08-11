import torch
import torch.nn as nn
import numpy as np
import cv2
import os
import torchvision


class OpenEyesClassifier(nn.Module):
    def __init__(self):
        super(OpenEyesClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding='same')
        self.leaky_relu1 = nn.LeakyReLU(0.1)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.25)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding='same')
        self.leaky_relu2 = nn.LeakyReLU(0.1)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(0.25)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding='same')
        self.leaky_relu3 = nn.LeakyReLU(0.1)
        self.max_pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout(0.4)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1152, 128) 
        self.leaky_relu4 = nn.LeakyReLU(0.1)
        self.dropout4 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)
        # self.logsoftmax = nn.LogSoftmax(dim=-1)

        self.transform = torch.nn.Sequential(
                torchvision.transforms.Resize((24,24)),
                torchvision.transforms.Grayscale(),
                torchvision.transforms.Normalize(0.5, 0.5)
            )
        
        self.model_path = './open_eyes_classifier.pth'
        if os.path.exists(self.model_path):
            self.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
            print(f"Loaded model weights from {self.model_path}")



    def forward(self, x):
        x = self.leaky_relu1(self.conv1(x))
        x = self.max_pool1(x)
        x = self.dropout1(x)

        x = self.leaky_relu2(self.conv2(x))
        x = self.max_pool2(x)
        x = self.dropout2(x)

        x = self.leaky_relu3(self.conv3(x))
        x = self.max_pool3(x)
        x = self.dropout3(x)

        x = self.flatten(x)
        x = self.leaky_relu4(self.fc1(x))
        x = self.dropout4(x)

        # x = self.sigmoid(self.fc2(x))
        x = self.fc2(x)
        # return self.logsoftmax(x)
        return self.sigmoid(x)
        # return self.softmax(x) 

    def predict(self, inpIm):
        image = cv2.imread(inpIm)
        image = torch.from_numpy(image).permute(2,0,1)/255

        image = self.transform(image)
        self.eval()
        with torch.no_grad():
            output = self.forward(image.unsqueeze(0))

        is_open_score = output.item()
        return is_open_score

