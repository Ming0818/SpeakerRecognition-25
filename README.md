*Path config* is in file: `PathConfig.py`

There are some utils:
- `enhance_speech.py`: wave denoise, using Spectral Subtraction method.
- `get_mfcc.py`: extract mfcc features.
- `read_wave.py`: read wave file.
- `nextpow2.py`: return the closest one(x) to 2^(x).


# 2018 未来杯高校AI挑战赛 决赛参赛作品

* 战队编号：{200225}
* 战队名称: {winner}
* 战队成员：{周扬、陈勇、陈发兵、杨凯、盛正兴}

## 概述

使用谱减法对语音降噪，提取MFCC特征，经过lstm层和若干全连层，两两比较，得出预测结果。

## 系统要求

windows10

### 硬件环境要求

* 本地运行CPU:i7-7800X
* 本地运行GPU:1080Ti
* 本地运行内存:32G
* 本地运行硬盘:
* 本地运行其他:

### 软件环境要求

* 操作系统: {windows} {10}
* {keras} {2.2.0}
* {tensorflow-gpu} {1.8.0}

如有特殊编译/安装步骤，请列明。

### 数据集

如使用官方提供的数据集之外的数据，请在此说明。

## 数据预处理

### 方法概述

先在`OtUtils/PathConfig.py`文件中配置好相关路径

运行`gain_label.py`获取标签文件

使用谱减法对语音降噪。然后提取MFCC特征并进行切片、归一化处理

### 操作步骤

运行`enhance.py`

## 训练

### 训练方法概述

提取语音信号MFCC特征并进行切片、归一化处理，再依照事先生成的标签文件输入数据进行训练

### 训练操作步骤

执行`Model/mfcc-lstm/mfcc-lstm.py`文件开始训练

### 训练结果保存与获取

训练模型保存在训练Python文件同一路径下。

## 测试

### 方法概述

先将需要测试的源文件进行降噪处理，执行`Model/mfcc-lstm/predict.py`中的`enhance`函数（只需执行一次即可），然后执行其中的`save_result`函数开始测试并将结果保存在`./res/`目录中

### 操作步骤

直接运行`Model/mfcc-lstm/predict.py`文件
