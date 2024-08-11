import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
from tqdm import tqdm
import gc
from autoencoder import Autoencoder
from utils import *
from torch.utils.tensorboard import SummaryWriter
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




def train(model, n_epochs, train_loader, val_loader, optimizer, criterion, exp_folder, save_each=5, writer=None):
    log_interval = 100
    num_imgs_plot = 6
    seed = 11
    torch.manual_seed(seed)

    train_loss = []
    val_loss = []

    for epoch in tqdm(range(1, n_epochs+1)):
        reconstructions_list = []
        reconstructions_val_list = []
        model.train()
        running_loss = 0.0
        running_val_loss = 0.0
        for idx, batch in enumerate(train_loader):
            idc = torch.randperm(batch.shape[0])[:num_imgs_plot]
            batch = batch.permute(0, 3, 1, 2).to(device).to(torch.float32)/255
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch)
            loss.backward()
            optimizer.step()

            if writer:
                writer.add_scalar("Loss/train", loss.item(), epoch)

            reco = output[idc].detach().cpu()
            reconstructions_list.append(reco)

            if idx % log_interval == 0:

                orig = batch[idc].detach().cpu()
                fig_reco = get_figure(reco, f'Reconstructed images for {epoch} || {idx}/{len(train_loader)}')
                fig_orig = get_figure(orig, f'Original images for {epoch} || {idx}/{len(train_loader)}')
                
                if writer:
                    writer.add_figure('Reconstructed images/train', fig_reco, epoch*len(train_loader) + idx)
                    plt.close()

                    writer.add_figure('Original images/train', fig_orig, epoch*len(train_loader) + idx)
                    plt.close()

                print(f"\nEpoch: {epoch} || {idx}/{len(train_loader)}\tLoss: {loss.item()}")

                del orig, fig_reco, fig_orig
            del reco
            gc.collect()


        model.eval()
        with torch.no_grad():
            for id, batch in enumerate(val_loader):
                idc = torch.randperm(batch.shape[0])[:num_imgs_plot]
                batch = batch.permute(0, 3, 1, 2).to(device).to(torch.float32)/255

                output = model(batch)
                loss = criterion(output, batch)

                val_loss.append(loss.item())
                writer.add_scalar("Loss/val", loss.item(), epoch)

                reco = output[idc].detach().cpu()

                reconstructions_val_list.append(reco)

                if id % log_interval == 0:
                    orig = batch[idc].detach().cpu()
                    fig_reco_val = get_figure(reco, f'Validation Reconstructed images for {epoch} || {id}/{len(val_loader)}')
                    fig_orig_val = get_figure(orig, f'Validation Original images for {epoch} || {id}/{len(val_loader)}')
                    
                    if writer:
                        writer.add_figure('Reconstructed images/val', fig_reco_val, epoch*len(val_loader) + id)
                        plt.close()

                        writer.add_figure('Original images/val', fig_orig_val, epoch*len(val_loader) + id)
                        plt.close()

                    print(f"\nEpoch: {epoch} || {id}/{len(val_loader)}\tValidation Loss: {loss.item()}")

                del orig, fig_reco_val, fig_orig_val
                del reco
                gc.collect()

            if epoch % save_each == 0:
                torch.save(model.state_dict(), f'{exp_folder}/model_epoch_{epoch}.pth')
                print(f"\nModel saved at epoch {epoch}")

    torch.save(model.state_dict(), f'{exp_folder}/autoencoder.pth')
    writer.flush()
    return f'{exp_folder}/autoencoder.pth'



def test_autoencoder(saved_model, test_loader, savedir):
    autoencoder_loaded = Autoencoder()
    autoencoder_loaded.load_state_dict(torch.load(saved_model))

    autoencoder_loaded.eval()
    reco_list = []
    orig_list = []
    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            data = data.permute(0, 3, 1, 2).to(device).to(torch.float32)/255
            recon = autoencoder_loaded(data)

            reco = recon.detach().cpu()
            orig = data.detach().cpu()
            reco_list.append(reco)
            orig_list.append(orig)

    fig_orig_test = get_figure(orig, f'Test Original images')
    fig_reco_test = get_figure(reco_list[-1], f'Test Reconstructed images')

    plt.savefig(f'{savedir}/oroginal_imgs.png', fig_orig_test)
    plt.savefig(f'{savedir}/reconstructed_imgs.png', fig_reco_test)



def main():
    all_images = get_images('./data/EyesDataset/')
    train_photos, temp_photos = train_test_split(all_images, train_size=0.8, shuffle=True, random_state=42)

    val_photos, test_photos = train_test_split(temp_photos, train_size=0.5, shuffle=True, random_state=42)

    train_loader = torch.utils.data.DataLoader(train_photos, batch_size=32, pin_memory=True, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_photos, batch_size=32, pin_memory=True, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_photos, batch_size=32, pin_memory=True, shuffle=False)

    print(f"Train size: {len(train_photos)}")
    print(f"Validation size: {len(val_photos)}")
    print(f"Test size: {len(test_photos)}")


    n_epochs = 100
    model = Autoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    writer = SummaryWriter("run/")

    saved_model_path = train(model, n_epochs, train_loader, val_loader, optimizer, criterion, 'models/exp1', save_each=10, writer=None)
    test_autoencoder(saved_model_path, test_loader, 'figs/exp1')




if __name__ == '__main__':
    main()