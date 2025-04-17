import torch

from utils.prepare_dataset import prepare
from utils.models import My_model


def apply(model_path, data_path, prepared_data_path, statistic_data_path, gray_image_path, tch_image_path):
    prepared_data_dict = prepare(data_path, prepared_data_path, gray_image_path, tch_image_path)
    statistic_data = torch.Tensor(prepared_data_dict['statistic_data']).float()
    gray_latent = torch.Tensor(prepared_data_dict['gray_latent']).float().squeeze(1)
    tch_latent = torch.Tensor(prepared_data_dict['tch_latent']).float().squeeze(1)

    statistic_flat = statistic_data.float().view(statistic_data.size(0), -1)
    features = torch.cat((statistic_flat, gray_latent, tch_latent), dim=1)

    model = My_model()
    model.load_state_dict(torch.load(model_path))

    logit = model(features)
    pred = torch.argmax(logit, dim=1).item()

    if pred == 1:
        print('Welcome!')
    else:
        print("Sorry!")


if __name__ == '__main__':
    data_path = 'path/to/writing/inertial_signal'

    prepared_data_path = 'path/to/prepared_data'

    gray_image_path = 'path/to/grayscale_image'

    tch_image_path = 'path/to/trajectory_image'

    model_path = 'path/to/classifier/model'

    apply(model_path, data_path, prepared_data_path, gray_image_path, tch_image_path)
