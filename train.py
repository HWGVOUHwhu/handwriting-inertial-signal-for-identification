import torch
import os
from torch import optim, nn
from utils.train_loop import train_loop
from utils.dataset_utils import get_loader
from utils.models import My_model


def train(data_folder, model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs = 300

    model = My_model().to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))

    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)
    loss_fn = nn.CrossEntropyLoss()

    print('preparing dataset...')
    train_loader = get_loader(data_folder, 'train')
    test_loader = get_loader(data_folder, 'test')

    print('training...')
    train_losses, test_losses, train_acc, test_acc = train_loop(device, optimizer, scheduler, loss_fn, epochs, train_loader, test_loader, model, model_path)


if __name__ == '__main__':
    model_path = r'models/My_model.pth'

    data_folder = 'dataset'

    train(data_folder, model_path)
