import numpy as np
import torch
from torchvision import transforms
from PIL import Image

from utils.preprocess import process
from utils.draw import draw_gray, draw_3ch
from utils.statistic import get_statistic
from utils.models import Conv_VAE
import os

data_folder = 'dataset'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

filterd_dataset_folder = 'f_dataset'
os.makedirs(filterd_dataset_folder, exist_ok=True)

statistic_dataset_folder = 'statistic'
os.makedirs(statistic_dataset_folder, exist_ok=True)

gray_image_folder = 'gray_images'
os.makedirs(gray_image_folder, exist_ok=True)

tch_image_folder = '3ch_images'
os.makedirs(tch_image_folder, exist_ok=True)

gray_model_path = r'D:\progrems\My_work_last\models\gray_VAE.pth'
gray_model = Conv_VAE().to(device)
gray_model.load_state_dict(torch.load(gray_model_path))
gray_model.eval()

tch_model_path = r'D:\progrems\My_work_last\models\3ch_VAE.pth'
tch_model = Conv_VAE(in_ch=3).to(device)
tch_model.load_state_dict(torch.load(tch_model_path))
tch_model.eval()

tf_color = transforms.Compose([
    lambda x:Image.open(x).convert('RGB'),
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor()
])

tf_gray = transforms.Compose([
    lambda x:Image.open(x).convert('RGB'),
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])


def prepare(data_path, preprocessed_data_path, statistic_data_path, gray_image_path, tch_image_path):
    process(data_path, preprocessed_data_path)
    get_statistic(preprocessed_data_path, statistic_data_path)
    draw_gray(preprocessed_data_path, gray_image_path)
    draw_3ch(preprocessed_data_path, tch_image_path)

    with torch.no_grad():
        gray_image = tf_gray(gray_image_path)
        gray_image = gray_image.to(device).unsqueeze(0)
        gray_latent = gray_model.get_latent(gray_image)

        tch_image = tf_color(tch_image_path)
        tch_image = tch_image.to(device).unsqueeze(0)
        tch_latent = tch_model.get_latent(tch_image)

    data = np.loadtxt(preprocessed_data_path)
    statistic_data = np.loadtxt(statistic_data_path)

    return {'data': data, 'statistic_data': statistic_data, 'gray_latent': gray_latent, 'tch_latent': tch_latent}


if __name__ == '__main__':
    for folder in os.listdir(data_folder):
        os.makedirs(os.path.join(filterd_dataset_folder, folder), exist_ok=True)
        os.makedirs(os.path.join(statistic_dataset_folder, folder), exist_ok=True)
        os.makedirs(os.path.join(gray_image_folder, folder), exist_ok=True)
        os.makedirs(os.path.join(tch_image_folder, folder), exist_ok=True)

        folder_path = os.path.join(data_folder, folder)
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)

            filterd_dataset_path = os.path.join(filterd_dataset_folder, folder, file)
            statistic_dataset_path = os.path.join(statistic_dataset_folder, folder, file)
            gray_image_path = os.path.join(gray_image_folder, folder, f"{file.split('.')[0]}.jpg")
            tch_image_path = os.path.join(tch_image_folder, folder, f"{file.split('.')[0]}.jpg")

            _ = prepare(file_path, filterd_dataset_path, statistic_dataset_path, gray_image_path, tch_image_path)

