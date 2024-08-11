import os
import cv2
from torch.utils.data import Dataset
import torch
import torch.nn as nn
from PIL import Image
import torchvision
import random


class FunctionalModule(torch.nn.Module):
    def __init__(self, functional):
        super().__init__()
        self.functional = functional

    def forward(self, input):
        return self.functional(input)


class ImageTransform():
    def __init__(self, subset):
        if subset == "train":
            self.image_pipeline = torch.nn.Sequential(
                torchvision.transforms.Resize((24,24)),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomRotation(10),
                torchvision.transforms.Grayscale(),
                torchvision.transforms.Normalize(0.5, 0.5)
                # torchvision.transforms.ToTensor()

            )
        elif subset == "val" or subset == "test":
            self.image_pipeline = torch.nn.Sequential(
                torchvision.transforms.Resize((24,24)),
                torchvision.transforms.Grayscale(),
                torchvision.transforms.Normalize(0.5, 0.5)
                # torchvision.transforms.ToTensor()
            )

    def __call__(self, sample):
        # sample: B x C x H x W
        # print(sample.shape)
        return self.image_pipeline(sample)



class EyeDataset(Dataset):
    def __init__(self, images, labels, transform=None):

        # self.open_images = [os.path.join(open_dir, img) for img in os.listdir(open_dir) if img.endswith(('png', 'jpg'))]
        # self.close_images = [os.path.join(close_dir, img) for img in os.listdir(close_dir) if img.endswith(('png', 'jpg'))]
        # self.images = self.open_images + self.close_images
        # self.labels = [1] * len(self.open_images) + [0] * len(self.close_images)
        self.images = images.copy()
        self.labels = labels.copy()
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        # image = Image.open(image_path)
        image = cv2.imread(image_path)
        # print(image)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # image = cv2.resize(image, (24, 24))
        image = torch.from_numpy(image).permute(2,0,1)/255
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        # return image, label
        return {'image': image, 'label': torch.tensor(label).float()} 