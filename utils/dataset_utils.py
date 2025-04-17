import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from utils.prepare_dataset import prepare
from sklearn.model_selection import train_test_split

tf_color = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor()
])

tf_gray = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])


class ImageSet(Dataset):
    def __init__(self, data_dir, mode, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.mode = mode
        self.image_paths = []
        self._prepare_dataset()

    def _prepare_dataset(self):
        for folder in os.listdir(self.data_dir):
            count = 0
            folder_path = os.path.join(self.data_dir, folder)
            for image_path in os.listdir(folder_path):
                count += 1
                if (self.mode == 'train') & (count <= 8):
                    self.image_paths.append(os.path.join(folder_path, image_path))
                if (self.mode == 'test') & (count > 8):
                    self.image_paths.append(os.path.join(folder_path, image_path))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image


def get_image_loader(data_dir, mode, batch_size, color='3ch'):
    if color == 'gray':
        dataset = ImageSet(data_dir, mode, tf_gray)
    else:
        dataset = ImageSet(data_dir, mode, tf_color)
    loader = DataLoader(dataset, batch_size, shuffle=False, num_workers=0)
    return loader


filterd_dataset_folder = 'f_dataset'
os.makedirs(filterd_dataset_folder, exist_ok=True)

statistic_dataset_folder = 'statistic'
os.makedirs(statistic_dataset_folder, exist_ok=True)

gray_image_folder = 'gray_images'
os.makedirs(gray_image_folder, exist_ok=True)

tch_image_folder = '3ch_images'
os.makedirs(tch_image_folder, exist_ok=True)


class Data_set(Dataset):
    def __init__(self, data_dir, mode):
        self.data_dir = data_dir
        self.mode = mode

        self.filtered_data = []
        self.statistic_data = []
        self.gray_latent = []
        self.tch_latent = []
        self.labels = []

        self.data = self.get_data(self.mode)

        for item in self.data:
            self.filtered_data.append(item['data'])
            self.statistic_data.append(item['statistic_data'])
            self.gray_latent.append(item['gray_latent'])
            self.tch_latent.append(item['tch_latent'])
            self.labels.append(item['label'])

    def get_data(self, mode):
        all_dict = []

        for folder in os.listdir(self.data_dir):
            os.makedirs(os.path.join(filterd_dataset_folder, folder), exist_ok=True)
            os.makedirs(os.path.join(statistic_dataset_folder, folder), exist_ok=True)
            os.makedirs(os.path.join(gray_image_folder, folder), exist_ok=True)
            os.makedirs(os.path.join(tch_image_folder, folder), exist_ok=True)

            folder_path = os.path.join(self.data_dir, folder)
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                filterd_dataset_path = os.path.join(filterd_dataset_folder, folder, file)
                statistic_dataset_path = os.path.join(statistic_dataset_folder, folder, file)
                gray_image_path = os.path.join(gray_image_folder, folder, f"{file.split('.')[0]}.jpg")
                tch_image_path = os.path.join(tch_image_folder, folder, f"{file.split('.')[0]}.jpg")

                dict = prepare(file_path, filterd_dataset_path, statistic_dataset_path, gray_image_path, tch_image_path)
                dict['label'] = int(folder)
                all_dict.append(dict)

        train_data, test_data = train_test_split(all_dict, test_size=0.2, random_state=42, shuffle=True)

        if mode == 'train':
            return train_data
        elif mode == 'test':
            return test_data
        else:
            raise ValueError('Mode Error!')

    def __len__(self):
        return len(self.gray_latent)

    def __getitem__(self, index):
        filtered_data = self.filtered_data[index]
        statistic_data = self.statistic_data[index]
        gray_latent = self.gray_latent[index]
        tch_latent = self.tch_latent[index]
        label = self.labels[index]

        return filtered_data, statistic_data, gray_latent, tch_latent, label


def get_loader(data_dir, mode):
    data_set = Data_set(data_dir, mode)
    loader = DataLoader(data_set, batch_size=10, shuffle=False, num_workers=0)
    return loader
