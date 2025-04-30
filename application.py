import torch
import numpy as np
from utils.prepare_dataset import prepare
from utils.models import My_model


def apply(model_path, data_path, prepared_data_path, statistic_data_path, gray_image_path, tch_image_path):
    prepared_data_dict = prepare(data_path, prepared_data_path, statistic_data_path, gray_image_path, tch_image_path)
    
    
    # 确保所有张量都是扁平的一维向量
    statistic_data = torch.tensor(prepared_data_dict['statistic_data'], dtype=torch.float32).flatten()
    gray_latent = torch.tensor(prepared_data_dict['gray_latent'].cpu().numpy(), dtype=torch.float32).flatten()
    tch_latent = torch.tensor(prepared_data_dict['tch_latent'].cpu().numpy(), dtype=torch.float32).flatten()
    
    # 将所有特征连接成一个向量并添加批次维度
    features = torch.cat((statistic_data, gray_latent, tch_latent), dim=0).unsqueeze(0)
    
    # 加载模型并进行预测
    model = My_model()
    model.load_state_dict(torch.load(model_path))
    model.eval()  # 设置为评估模式
    
    with torch.no_grad():  # 在评估时不需要计算梯度
        logit = model(features)
        pred = torch.argmax(logit, dim=1).item()
    
    if pred == 1:
        print('Welcome!')
    else:
        print("Sorry!")


if __name__ == '__main__':
    data_path = r'dataset/1/2.txt'
    prepared_data_path = r'f_dataset/1/1.txt'
    statistic_data_path = r'statistic/1/1.txt'
    gray_image_path = r'gray_images/1/1.jpg'
    tch_image_path = r'3ch_images/1/1.jpg'
    model_path = r'models/My_model.pth'

    apply(model_path, data_path, prepared_data_path, statistic_data_path, gray_image_path, tch_image_path)