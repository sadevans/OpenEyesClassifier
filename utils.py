import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import random
import torch
from sklearn.model_selection import train_test_split





def extract_features(model, data_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    features = []
    model.eval()
    with torch.no_grad(): 
        for images in data_loader:
            data = images.permute(0, 3, 1, 2).to(device).to(torch.float32)/255
            encoded_features = model.encoder(data)
            features.append(encoded_features.reshape(encoded_features.size(0), -1))
    return torch.cat(features, dim=0)


def get_figure(images, title):

  fig, axes = plt.subplots(3, 4, figsize=(12,8))
  fig.suptitle(title)
  for i, ax in enumerate(axes.flat):
    if i<len(images):
      ax.imshow((np.transpose(images[i].detach().cpu().numpy(), (1, 2, 0))*255).astype('uint8'))

    ax.axis('off')

  plt.tight_layout()
  plt.subplots_adjust(top=0.9)
  return fig


def get_images(dataset_dir):
    paths = os.listdir(dataset_dir)
    all_photos = []
    for path in paths:
      img = cv2.imread(os.path.join(dataset_dir, path))
      all_photos.append(img)
    return all_photos


def show_examples(train_X, train_Y):
    print()
    print('Пример размеченных изображений:')
    class_names = ['Closed', 'Opened']
    plt.figure(figsize=(12, 10))
    for i in range(15):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        index = random.randint(0, len(train_X))
        plt.imshow(train_X[index], cmap=plt.cm.gray)
        plt.xlabel(class_names[int(train_Y[index])])
    plt.show()


def load_image_paths_and_labels(open_dir, close_dir):
    open_images = [os.path.join(open_dir, img) for img in os.listdir(open_dir) if img.endswith(('png', 'jpg', 'jpeg'))]
    close_images = [os.path.join(close_dir, img) for img in os.listdir(close_dir) if img.endswith(('png', 'jpg', 'jpeg'))]
    images = open_images + close_images
    labels = [1] * len(open_images) + [0] * len(close_images)
    return images, labels


def shuffle_data(images, labels):
    combined = list(zip(images, labels))
    random.shuffle(combined)
    images[:], labels[:] = zip(*combined)
    return images, labels


def split_data(images, labels, seed):
    train_images, temp_images, train_labels, temp_labels = train_test_split(images, labels, test_size=0.2, stratify=labels, random_state=seed)
    val_images, test_images, val_labels, test_labels = train_test_split(temp_images, temp_labels, test_size=0.5, stratify=temp_labels, random_state=seed)
    return train_images, val_images, test_images, train_labels, val_labels, test_labels
