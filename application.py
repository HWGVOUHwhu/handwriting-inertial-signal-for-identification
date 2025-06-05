import torch
import numpy as np
from utils.prepare_dataset import prepare
from utils.models import My_model
import random


def apply(
    model_path,
    data_path,
    prepared_data_path,
    statistic_data_path,
    gray_image_path,
    tch_image_path,
):
    """
    结合下面的main函数，感觉这里有问题，prepare是data_path对应的路径放到其他处理后的特征路径，
    但是此处data_path与其他特征路径后缀不一致，相当于把其他数据覆盖了
    """
    prepared_data_dict = prepare(
        data_path,
        prepared_data_path,
        statistic_data_path,
        gray_image_path,
        tch_image_path,
    )

    # 确保所有张量都是扁平的一维向量
    statistic_data = torch.tensor(
        prepared_data_dict["statistic_data"], dtype=torch.float32
    ).flatten()
    gray_latent = torch.tensor(
        prepared_data_dict["gray_latent"].cpu().numpy(), dtype=torch.float32
    ).flatten()
    tch_latent = torch.tensor(
        prepared_data_dict["tch_latent"].cpu().numpy(), dtype=torch.float32
    ).flatten()

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
        print("Welcome!")
    else:
        print("Sorry!")


def accuracy_test(model_path, test_num):
    """
    随机选取样本，批量计算模型在随机样本上的准确率。

    参数:
    model_path (str): 模型文件的路径。
    n (int): 用于测试的样本数量。

    返回:
    None: 直接打印准确率。
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    samples = np.random.randint(low=0, high=2, size=test_num)

    # 加载模型并进行预测
    model = My_model()
    model.load_state_dict(torch.load(model_path))
    model.eval()  # 设置为评估模式

    preds = []  # 储存预测结果对象
    all_features = []  # 存储所有样本的特征

    for i in range(test_num):
        partial_path = str(samples[i]) + r"/" + str(random.randint(1, 10))
        preprocessed_data_path = r"f_dataset/" + partial_path + ".txt"
        statistic_data_path = r"statistic/" + partial_path + ".txt"
        gray_image_path = r"gray_images/" + partial_path + ".jpg"
        tch_image_path = r"3ch_images/" + partial_path + ".jpg"
        prepared_data_dict = prepare(
                data_path = '',
                preprocessed_data_path=preprocessed_data_path,
                statistic_data_path=statistic_data_path,
                gray_image_path=gray_image_path,
                tch_image_path=tch_image_path,
                _process_ = False
            )

        # 确保所有张量都是扁平的一维向量
        statistic_data = torch.tensor(
            prepared_data_dict["statistic_data"], dtype=torch.float32
        ).flatten()
        gray_latent = torch.tensor(
            prepared_data_dict["gray_latent"].cpu().numpy(), dtype=torch.float32
        ).flatten()
        tch_latent = torch.tensor(
            prepared_data_dict["tch_latent"].cpu().numpy(), dtype=torch.float32
        ).flatten()

        # 将所有特征连接成一个向量并添加批次维度
        features = torch.cat(
            (statistic_data, gray_latent, tch_latent), dim=0
        )
        all_features.append(features)

    # 将所有样本的特征合并成一个批次
    all_features = torch.stack(all_features).to(device)

    with torch.no_grad():  # 在评估时不需要计算梯度
        logits = model(all_features)
        preds = torch.argmax(logits, dim=1).cpu().numpy()   # preds列表中的元素表示预测的对象，例如0表示预测数据来自对象0
    
    correct = (preds == samples).astype(int)
    accuracy = correct.sum() / test_num * 100
    print(f"准确率为 {accuracy}%")


if __name__ == '__main__':
    # data_path = r'dataset/1/2.txt'
    # prepared_data_path = r'f_dataset/1/1.txt'
    # statistic_data_path = r'statistic/1/1.txt'
    # gray_image_path = r'gray_images/1/1.jpg'
    # tch_image_path = r'3ch_images/1/1.jpg'
    # model_path = r'models/My_model.pth'

    # apply(model_path, data_path, prepared_data_path, statistic_data_path, gray_image_path, tch_image_path)

    model_path = r'models/My_model.pth'
    n = 20
    accuracy_test(model_path, n)