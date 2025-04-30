import torch
import os
from torch import optim
from tqdm import tqdm
from utils.models import Conv_VAE
from utils.dataset_utils import get_image_loader
import torch.nn.functional as F


def train_loop(model, optimizer, scheduler, device, loss_fn, epochs, train_loader, test_loader, model_path):
    train_losses, test_losses = [], []
    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            image = batch
            image = image.to(device)

            recon, mu, logvar = model(image)
            loss = loss_fn(recon, image, mu, logvar)
            # print(f"recon:{recon}, mu:{mu}, logvar:{logvar}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() / len(train_loader)

        train_losses.append(train_loss)

        scheduler.step()

        model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                image = batch
                image = image.to(device)

                recon, mu, logvar = model(image)
                loss = loss_fn(recon, image, mu, logvar)

                test_loss += loss.item() / len(test_loader)

        test_losses.append(test_loss)

        if test_loss < best_loss:
            os.makedirs(os.path.dirname(model_path), exist_ok=True);
            torch.save(model.state_dict(), model_path)
            best_loss = test_loss
            print('Saved best model!')

        print(
            f'Epoch-{epoch + 1}: train_loss-{train_loss} test_loss-{test_loss}')

    return train_losses, test_losses


def MSE_KL(recon_x, x, mu, logvar, beta=1.0):
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    logvar = torch.clamp(logvar, min=-5.0, max=5.0)
    kl_div = -0.5*torch.sum(1+logvar - mu.pow(2) - logvar.exp()) / x.size(0)

    return recon_loss + beta * kl_div


def train_gray_VAE(model_path, data_folder):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Conv_VAE().to(device)

    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))

    optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)
    loss_fn = MSE_KL
    epochs = 300
    batch_size = 2

    train_loader = get_image_loader(data_folder, 'train', batch_size, 'gray')
    test_loader = get_image_loader(data_folder, 'test', batch_size, 'gray')

    train_losses, test_losses = train_loop(model, optimizer, scheduler, device, loss_fn, epochs, train_loader, test_loader, model_path)


def train_3ch_VAE(model_path, data_folder):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Conv_VAE(in_ch=3).to(device)

    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))

    optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)
    loss_fn = MSE_KL
    epochs = 300
    batch_size = 2

    train_loader = get_image_loader(data_folder, 'train', batch_size)
    test_loader = get_image_loader(data_folder, 'test', batch_size)

    train_losses, test_losses = train_loop(model, optimizer, scheduler, device, loss_fn, epochs, train_loader, test_loader, model_path)


if __name__ == '__main__':
    model_path = 'models/gray_VAE.pth'
    data_folder = 'gray_images'
    print('training gray VAE...')
    train_gray_VAE(model_path, data_folder)

    model_path = 'models/3ch_VAE.pth'
    data_folder = '3ch_images'
    print('training 3ch VAE...')
    train_3ch_VAE(model_path, data_folder)
