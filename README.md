# 基于惯性信号的手写身份识别系统

本项目利用深度学习技术，基于手写过程中的惯性信号（加速度计和陀螺仪数据）进行身份识别。系统通过卷积变分自编码器（Conv-VAE）提取特征，并结合统计分析方法实现高准确率的身份识别。

## 项目架构

```
handwriting-inertial-signal-for-identification/
├── application.py              # 主应用程序接口
├── README.md                   # 项目说明文档
├── requirements.txt            # 项目依赖环境配置
├── train_VAE.py                # 卷积变分自编码器训练脚本
├── train.py                    # 身份识别模型训练脚本
├── dataset/                    # 原始数据集目录
│   ├── 0/                      # 用户0的数据(正例)
│   ├── 1/                      # 用户1的数据(负例)
│   └── ...                     # 更多用户数据
├── models/                     # 预训练模型
│   ├── 3ch_VAE.pth            # 三通道轨迹投影图-VAE模型
│   ├── gray_VAE.pth           # 灰度图-VAE模型
│   └── My_model.pth           # 身份识别模型
└── utils/                      # 工具函数
    ├── __init__.py
    ├── dataset_utils.py        # 数据集处理工具
    ├── draw.py                 # 可视化工具
    ├── models.py               # 模型定义
    ├── prepare_dataset.py      # 数据集预处理工具
    ├── preprocess.py           # 信号预处理
    ├── statistic.py            # 统计特征提取
    └── train_loop.py           # 训练循环
```

## 主要功能

本项目实现了以下主要功能：

1. **信号预处理**：对原始惯性信号进行滤波和对齐处理
2. **特征提取**：
   - 时域特征：均值、标准差、最大值、最小值、均方根
   - 频域特征：峰值频率、频谱能量
   - 视觉特征：通过卷积变分自编码器（Conv-VAE）提取
3. **数据可视化**：将惯性信号转换为灰度图和三通道RGB轨迹图
4. **身份识别**：基于提取的特征进行身份识别

## 技术实现

### 数据预处理

原始惯性信号通过Savitzky-Golay滤波器进行去噪，并通过样条插值统一到相同长度

### 特征提取

项目结合时域、频域和视觉特征进行分析：

1. **统计特征**：从时域和频域提取描述性统计量
2. **视觉特征**：
   - 将处理后的信号转换为灰度图像
   - 通过惯性信号计算轨迹生成三通道RGB图像
   - 使用Conv-VAE从图像中提取潜在特征

### 模型架构

#### 变分自编码器 (Conv_VAE)

用于从生成的图像中提取特征的卷积变分自编码器，架构包含：
- 编码器：6层卷积层，带BatchNorm和ReLU激活
- 解码器：6层转置卷积层
- 潜在空间：32维向量

#### 识别模型 (My_model)

用于身份识别的简单前馈神经网络：
- 输入：106维特征（包括统计特征和VAE潜在特征）
- 隐藏层：64个神经元，ReLU激活
- 输出：2类（二分类身份识别）

## 安装与使用

1. 克隆仓库：
   ```bash
   git clone https://github.com/yourusername/handwriting-inertial-signal-for-identification.git
   cd handwriting-inertial-signal-for-identification
   ```

2. 创建并激活虚拟环境：
   ```bash
   python -m venv isid
   source isid/bin/activate  # Linux/Mac
   # 或
   isid\Scripts\activate  # Windows
   ```

3. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

### 数据准备

1. 将惯性信号数据放置在 dataset 目录下，按用户ID组织文件夹结构
2. 运行数据预处理脚本：
   ```bash
   python -m utils.prepare_dataset
   ```

### 模型训练

1. 训练变分自编码器：
   ```bash
   python train_VAE.py
   ```

2. 训练身份识别模型：
   ```bash
   python train.py
   ```

### 应用运行

运行主应用程序：
```bash
python3 application.py
```

## 数据可视化示例

项目将惯性信号转换为两种可视化表示：

1. **灰度图像**：信号振幅的二维表示
2. **轨迹投影图**：三个平面（XY、XZ、YZ）上的轨迹投影，分别用RGB三个通道表示

## 实验结果

在小样本测试集上，本模型实现了高准确率（接近100%）的身份识别，详细实验结果请参考训练日志。

## 未来改进

1. 增加更多特征提取方法
2. 支持更多用户的多分类识别
3. 实现实时识别功能
4. 优化模型架构提高准确率
5. 在大规模样本集上测试判别准确性
