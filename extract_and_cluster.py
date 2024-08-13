import torch
import cv2
from utils import *
from autoencoder import Autoencoder
from sklearn.model_selection import train_test_split
import shutil

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def split_dataset(dataset_path, destination_path):
    all_images = os.listdir(dataset_path)
    train_photos, temp_photos = train_test_split(all_images, train_size=0.8, shuffle=True, random_state=42)

    val_photos, test_photos = train_test_split(temp_photos, train_size=0.5, shuffle=True, random_state=42)
    dict_ = {'train':train_photos, 'val':val_photos, 'test':test_photos}
    for split, photos in dict_.items():
        print(f'Started copying {split} images !')
        os.makedirs(os.path.join(destination_path, split), exist_ok=True)
        # if split=='train':
        for img in photos:
            image_path = os.path.join(dataset_path, img)
            shutil.copy(image_path, os.path.join(destination_path, split))
    print('Splitted all the data')


def organize_images_by_cluster(dataset_dir, image_paths, labels, output_dir="clustered_images"):
    """Organizes images into folders based on their cluster labels."""

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    zero_list = []
    one_list = []
    for i, label in enumerate(labels):
        cluster_dir = os.path.join(output_dir, f"cluster_{label}")
        if not os.path.exists(cluster_dir):
            os.makedirs(cluster_dir)
        
        image_path = os.path.join(dataset_dir, image_paths[i])
        shutil.copy(image_path, os.path.join(cluster_dir, os.path.basename(image_path)))

        if label == 0:
          zero_list.append(image_path)
        elif label == 1:
          one_list.append(image_path)
    return zero_list, one_list


def extract_and_cluster_imgs(images, saved_model_path, split, image_paths):
    all_dataloader = torch.utils.data.DataLoader(images, batch_size=32, shuffle=False)

    autoencoder_loaded = Autoencoder().to(device)
    autoencoder_loaded.load_state_dict(torch.load(saved_model_path))
    features_autoencoder = extract_features(autoencoder_loaded, all_dataloader)

    tsne = TSNE(n_components=2, random_state=42)
    reduced_features = tsne.fit_transform(features_autoencoder.cpu().numpy())

    kmeans_tcne = KMeans(n_clusters=2, random_state=42)
    clusters_tcne = kmeans_tcne.fit_predict(reduced_features)

    organize_images_by_cluster(f'./data/dataset/{split}', image_paths, clusters_tcne, output_dir=f"cluster_images/")





# def main(dataset_path, destination_path):

    

if __name__ == "__main__":
    # split_dataset('./data/EyesDataset', './data/dataset')
    # for split in ['train', 'val', 'test']:
    images = get_images(f'./data/dataset/EyesDataset')
    image_paths = os.listdir(f'./data/dataset/EyesDataset')
    extract_and_cluster_imgs(images, './models/autoencoder.pth', image_paths)