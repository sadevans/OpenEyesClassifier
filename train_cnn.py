from sklearn.metrics import roc_curve, auc
import numpy as np
import torch
import torch.nn as nn
from dataset import ImageTransform, EyeDataset
from OpenEyesClassificator import OpenEyesClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from utils import *
import cv2
from torch.utils.data import Dataset, DataLoader
import random
import torch.optim as optim
import wandb
import os
# from pytorchtools import EarlyStopping



def compute_eer(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    frr = 1 - tpr
    abs_diffs = np.abs(fpr - frr)
    min_index = np.argmin(abs_diffs)
    eer = (fpr[min_index]+ frr[min_index])/2
    
    return eer


def compute_accuracy(preds, labels):
    correct = (preds == labels).sum().item()
    total = preds.shape[0]

    return correct/total


def plot_confusion_matrix(conf_matrix):
    fig, ax = plt.subplots()
    heatmap = ax.imshow(conf_matrix, cmap='Blues')
    ax.set_xticks(np.arange(conf_matrix.shape[1]))
    ax.set_yticks(np.arange(conf_matrix.shape[0]))
    ax.set_xticklabels(['Class 0', 'Class 1'])
    ax.set_yticklabels(['Class 0', 'Class 1'])
    plt.colorbar(heatmap)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')

    return heatmap


def plot_images(images, labels, scores, num_images=12):

    fig, axes = plt.subplots(num_images//4, num_images//3, figsize=(12,8))
    for i, ax in enumerate(axes.flat):
        image = images[i]
        label = labels[i]
        prediction = scores[i]
        if i<len(images):
            ax.imshow(image, cmap='gray')
            ax.set_title(f"Label: {label}, Prediction: {prediction:.4f}")
        ax.axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    return fig

    
def evaluate_model(model, criterion, val_loader, device=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    val_loss = 0.0

    val_labels_list = []
    val_preds_list = []
    with torch.no_grad():
        for it in val_loader:
            inputs, labels = it['image'].to(device), it['label'].to(device)
            outputs = model(inputs)

            loss = criterion(outputs.permute(1,0), labels.unsqueeze(0))

            val_loss += loss.item()

            val_labels_list += labels.unsqueeze(0).cpu().detach().numpy().tolist()
            val_preds_list += outputs.permute(1,0).cpu().detach().numpy().tolist()

        val_eer = compute_eer(val_labels_list[0], val_preds_list[0])
        preds = (torch.tensor(val_preds_list[0]) > 0.5).float()

        val_acc = compute_accuracy(preds, torch.tensor(val_labels_list[0]))

    return val_loss, val_acc, val_eer



def train_model(model, train_loader, val_loader, criterion, optimizer, ckpts_path, num_epochs=1, \
                scheduler=None, device=None, logger=False):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    min_eer = np.inf
    for epoch in range(num_epochs):

        model.train()
        train_loss = 0.0
        correct = 0.0
        total = 0.0

        train_labels_list = []
        train_preds_list = []
        
        for it in train_loader:
            inputs, labels = it['image'].to(device), it['label'].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.permute(1,0), labels.unsqueeze(0))

            loss.backward()
            optimizer.step()
            train_loss += loss.item()


            train_labels_list += labels.unsqueeze(0).cpu().detach().numpy().tolist()
            train_preds_list += outputs.permute(1,0).cpu().detach().numpy().tolist()


        train_eer = compute_eer(train_labels_list[0], train_preds_list[0])
        preds = (torch.tensor(train_preds_list[0]) > 0.5).float()

        train_acc = compute_accuracy(preds, torch.tensor(train_labels_list[0]))

        val_loss, val_acc, val_eer = evaluate_model(model, criterion, val_loader, device=device)

        if logger:
            wandb.log({"epoch": epoch, "train_loss": train_loss, "train_accuracy": train_acc, "train_eer": train_eer, \
                    "val_loss": val_loss, "val_accuracy": val_acc, "val_eer": val_eer})
        
        print(f"Epoch {epoch} || validation accuracy = {val_acc:.4f}, validation eer = {val_eer:.4f}")
        if scheduler:
            scheduler.step()
            # scheduler.step(val_eer)


        if val_eer < min_eer or val_eer < 0.01:
            min_eer = val_eer
            torch.save(model.state_dict(), os.path.join(ckpts_path, f"model_epoch{epoch}_eer{min_eer:.4f}.pth"))
            print(f"Saved model with validation accuracy = {val_acc:.4f} and eer = {val_eer:.4f}")


    torch.save(model.state_dict(), "./open_eyes_classifier.pth")
    return model, min_eer


def test_model(test_images, test_labels, plot=False, n_images=12, name=None):

    model = OpenEyesClassifier()
    test_acc = 0.0
    scores_list = []
    imgs_list = []

    for image_path, label in zip(test_images, test_labels):
        
        score = model.predict(image_path)
        scores_list.append(score)
        if plot:
            img = cv2.imread(image_path, 0)
            imgs_list.append(img)

    test_eer = compute_eer(test_labels, scores_list)
    preds = (torch.tensor(scores_list) > 0.5).float()
    test_acc = compute_accuracy(preds, torch.tensor(test_labels))

    if plot:
        if name is None:
            name='Testing_images'
        plot_images(imgs_list, test_labels, scores_list, num_images=n_images)
        plt.savefig(f'./figs/{name}.png')

    print(f"Testing accuracy = {test_acc:.4f}, testing eer = {test_eer:.4f}")






def get_dataloaders():
    train_dataset = EyeDataset(train_images, train_labels, transform=ImageTransform('train'))
    val_dataset = EyeDataset(val_images, val_labels, transform=ImageTransform('val'))
    test_dataset = EyeDataset(test_images, test_labels, transform=ImageTransform('test'))


    train_loader = DataLoader(train_dataset, batch_size=64, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, pin_memory=True, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, pin_memory=True, shuffle=False)

    return train_loader, val_loader, test_loader






if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    open_dir = '/home/sadevans/space/CloseEyesClassifier/data/clustered_auto_tcne_CHECKED/open'
    close_dir = '/home/sadevans/space/CloseEyesClassifier/data/clustered_auto_tcne_CHECKED/close'


    seed = 31
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


    images, labels = load_image_paths_and_labels(open_dir, close_dir)
    images, labels = shuffle_data(images, labels)
    train_images, val_images, test_images, train_labels, val_labels, test_labels = split_data(images, labels, seed=seed)

    print(val_labels)
    print(f"Train: {len(train_images)} images")
    print(f"Validation: {len(val_images)} images")
    print(f"Test: {len(test_images)} images")

    train_loader, val_loader, test_loader = get_dataloaders()
    

    model = OpenEyesClassifier().to(device)

    wandb.init(project="OpenEyes")


    save_path = f"exp/classifier_23/"
    os.makedirs(save_path, exist_ok=True)

    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=15)

    trained_model, min_val_eer = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=40, \
                ckpts_path=save_path, scheduler=scheduler, device=device, logger=True)
    
    print(f'\nMin eval EER = {min_val_eer:.4f}\n')
    # print(test_images)
    test_model(val_images, val_labels)
    test_model(test_images, test_labels)

