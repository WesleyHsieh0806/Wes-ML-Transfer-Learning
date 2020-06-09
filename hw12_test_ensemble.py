# -*- coding: utf-8 -*-
import math
import sys
import os
import time
import pandas as pd
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import torch.optim as optim
from torch.autograd import Function
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import models
# Original paper:https://arxiv.org/pdf/1505.07818.pdf
# 這一部分是把每一個label的圖片畫出來
torch.manual_seed(0)
if not os.path.isdir('./result/vgg16_512'):
    os.makedirs('./result/vgg16_512')


def no_axis_show(img, title='', cmap=None):
    # imshow, 縮放模式為nearest。
    fig = plt.imshow(img, interpolation='nearest', cmap=cmap)
    # 不要顯示axis。
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.title(title)


"""# Data Process

在這裡我故意將data用成可以使用torchvision.ImageFolder的形式，所以只要使用該函式便可以做出一個datasets。

transform的部分請參考以下註解。

# 一些細節

在一般的版本上，對灰階圖片使用RandomRotation使用```transforms.RandomRotation(15)```即可。但在colab上需要加上```fill=(0,)```才可運行。
在n98上執行需要把```fill=(0,)```拿掉才可運行。
"""


# 上面的canny則是在transform當中執行
source_transform = transforms.Compose([
    # 轉灰階: Canny 不吃 RGB。
    transforms.Grayscale(),
    # cv2 不吃 skimage.Image，因此轉成np.array後再做cv2.Canny
    transforms.Lambda(lambda x: cv2.Canny(np.array(x), 170, 300)),
    # 重新將np.array 轉回 skimage.Image
    transforms.ToPILImage(),
    # 水平翻轉 (Augmentation)
    transforms.RandomHorizontalFlip(),
    # 旋轉15度內 (Augmentation)，旋轉後空的地方補0
    # transforms.RandomRotation(15, fill=(0,)),
    transforms.RandomRotation(15),
    # color jitter可以調整亮度飽和度等等
    transforms.ColorJitter(brightness=(0.5, 1.5), contrast=(
        0.5, 1.5), saturation=(0.5, 1.5)),
    # 透視變換
    transforms.RandomPerspective(),
    # 最後轉成Tensor供model使用。
    transforms.ToTensor(),
])
target_transform = transforms.Compose([
    # 轉灰階: 將輸入3維壓成1維。
    transforms.Grayscale(),
    # 縮放: 因為source data是32x32，我們將target data的28x28放大成32x32。
    transforms.Resize((32, 32)),
    # 水平翻轉 (Augmentation)
    transforms.RandomHorizontalFlip(),
    # 旋轉15度內 (Augmentation)，旋轉後空的地方補0
    # transforms.RandomRotation(15, fill=(0,)),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=(0.5, 1.5), contrast=(
        0.5, 1.5), saturation=(0.5, 1.5)),
    #
    transforms.RandomPerspective(),
    # 最後轉成Tensor供model使用。
    transforms.ToTensor(),
])
test_transform = transforms.Compose([
    # 轉灰階: 將輸入3維壓成1維。
    transforms.Grayscale(),
    # 縮放: 因為source data是32x32，我們將target data的28x28放大成32x32。
    transforms.Resize((32, 32)),
    # 最後轉成Tensor供model使用。
    transforms.ToTensor(),
])
# source data就是有label得data
# target data就是這次主要task所要用到的data
source_dataset = ImageFolder(
    os.path.join(sys.argv[1], "train_data"), transform=source_transform)
target_dataset = ImageFolder(
    os.path.join(sys.argv[1], "test_data"), transform=target_transform)
test_dataset = ImageFolder(
    os.path.join(sys.argv[1], "test_data"), transform=test_transform)

source_dataloader = DataLoader(source_dataset, batch_size=32, shuffle=True)
target_dataloader = DataLoader(target_dataset, batch_size=32, shuffle=True)
# 之所以target dataset要用兩次是因為 train discriminator的時候也需要用到target data
test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

"""# Model

Feature Extractor: 典型的VGG-like疊法。

Label Predictor / Domain Classifier: MLP到尾。(Multilayer Perceptron)

相信作業寫到這邊大家對以下的Layer都很熟悉，因此不再贅述。
"""


class resnet34_FeatureExtractor(nn.Module):

    def __init__(self):
        super(resnet34_FeatureExtractor, self).__init__()
        # resnet34
        model = models.resnet34(pretrained=False)
        self.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3)
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = model.avgpool

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # print(x.shape)
        x = self.layer4(x)
        # print(x.shape)
        return x.squeeze()


class resnet34_LabelPredictor(nn.Module):

    def __init__(self):
        super(resnet34_LabelPredictor, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.ReLU(),

            nn.Linear(512, 10),
        )

    def forward(self, h):
        c = self.layer(h)
        return c


class vgg16_512_FeatureExtractor(nn.Module):

    def __init__(self):
        super(vgg16_512_FeatureExtractor, self).__init__()
        # vgg16
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        x = self.conv(x).squeeze()
        return x


class vgg16_512_LabelPredictor(nn.Module):

    def __init__(self):
        super(vgg16_512_LabelPredictor, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(512, 512),

            nn.ReLU(),

            nn.Linear(512, 512),

            nn.ReLU(),

            nn.Linear(512, 10),
        )

    def forward(self, h):
        c = self.layer(h)
        return c


class vgg16_512_2_FeatureExtractor(nn.Module):

    def __init__(self):
        super(vgg16_512_2_FeatureExtractor, self).__init__()
        # vgg16
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        x = self.conv(x).squeeze()
        return x


class vgg16_512_2_LabelPredictor(nn.Module):

    def __init__(self):
        super(vgg16_512_2_LabelPredictor, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(512, 512),

            nn.ReLU(),

            nn.Linear(512, 512),

            nn.ReLU(),

            nn.Linear(512, 10),
        )

    def forward(self, h):
        c = self.layer(h)
        return c


"""# Pre-processing

這裡我們選用Adam來當Optimizer。
"""


class_criterion = nn.CrossEntropyLoss()
domain_criterion = nn.BCEWithLogitsLoss()

"""# Start Training


# 如何實作DaNN?

理論上，在原始paper中是加上Gradient Reversal Layer，並將Feature Extractor / Label Predictor / Domain Classifier 一起train，但其實我們也可以交換的train Domain Classfier & Feature Extractor(就像在train GAN的Generator & Discriminator一樣)，這也是可行的。

在code實現中，我們採取後者的方式，畢竟大家上個作業就是GAN，應該會比較熟悉:)。

# 小提醒
* 原文中的lambda(控制Domain Adversarial Loss的係數)是有Adaptive的版本，如果有興趣可以參考[原文](https://arxiv.org/pdf/1505.07818.pdf)。以下為了方便固定設置0.1。
* 因為我們完全沒有target的label，所以結果如何，只好丟kaggle看看囉:)?
"""


# 訓練200 epochs
best_loss = np.inf
best_acc = 0.
feature_extractor_loss = []
domain_loss = []
acc = []
num_epoch = 1500
# gamma用在計算lamb(Adaptation parameter)
gamma = 10

"""# Inference

就跟前幾次作業一樣。這裡我使用pd來生產csv，因為看起來比較潮(?)

此外，200 epochs的Accuracy可能會不太穩定，可以多丟幾次或train久一點。
"""
print("Let's do some big shit by Ensemble..")
result = []
# Model 1 Vgg16
vgg16_512_feature_extractor = vgg16_512_FeatureExtractor()
vgg16_512_label_predictor = vgg16_512_LabelPredictor()
print("Total # of parameters of Model 1 Feature Extractor:{}".format(
    sum(p.numel() for p in vgg16_512_feature_extractor.parameters())))
print("Total # of parameters of Model 1 label Predictor:{}".format(
    sum(p.numel() for p in vgg16_512_label_predictor.parameters())))
vgg16_512_feature_extractor.load_state_dict(torch.load(
    './result/vgg16_512/vgg16_extractor_model.bin'))
vgg16_512_label_predictor.load_state_dict(torch.load(
    './result/vgg16_512/vgg16_predictor_model.bin'))
vgg16_512_feature_extractor.eval()
vgg16_512_label_predictor.eval()
# Model2 Resnet34
resnet34_feature_extractor = resnet34_FeatureExtractor()
resnet34_label_predictor = resnet34_LabelPredictor()
print("\nTotal # of parameters of Model 2 Feature Extractor:{}".format(
    sum(p.numel() for p in resnet34_feature_extractor.parameters())))
print("Total # of parameters of Model 2 label Predictor:{}".format(
    sum(p.numel() for p in resnet34_label_predictor.parameters())))
resnet34_feature_extractor.load_state_dict(torch.load(
    './result/resnet34/resnet34_extractor_model.bin'))
resnet34_label_predictor.load_state_dict(torch.load(
    './result/resnet34/resnet34_predictor_model.bin'))
resnet34_feature_extractor.eval()
resnet34_label_predictor.eval()
# Model3: Vgg16
# vgg16_512_2_feature_extractor = vgg16_512_2_FeatureExtractor()
# vgg16_512_2_label_predictor = vgg16_512_2_LabelPredictor()

# print("\nTotal # of parameters of Model 3 Feature Extractor:{}".format(
#     sum(p.numel() for p in vgg16_512_2_feature_extractor.parameters())))
# print("Total # of parameters of Model 3 label Predictor:{}".format(
#     sum(p.numel() for p in vgg16_512_2_label_predictor.parameters())))
# vgg16_512_feature_extractor.load_state_dict(torch.load(
#     './result/vgg16_512_2/vgg16_extractor_model.bin'))
# vgg16_512_label_predictor.load_state_dict(torch.load(
#     './result/vgg16_512_2/vgg16_predictor_model.bin'))
# vgg16_512_2_feature_extractor.eval()
# vgg16_512_2_label_predictor.eval()

for i, (test_data, _) in enumerate(test_dataloader):
    test_data = test_data.cuda()

    # Ensemble part1:vgg16_512 acc:0.801
    vgg16_512_feature_extractor.cuda()
    vgg16_512_label_predictor.cuda()
    class_logits = vgg16_512_label_predictor(
        vgg16_512_feature_extractor(test_data))
    soft_logits1 = nn.functional.softmax(class_logits, dim=1)
    vgg16_512_feature_extractor.cpu()
    vgg16_512_label_predictor.cpu()
    # Ensemble Part2: Resnet34  acc:0.76918
    resnet34_feature_extractor.cuda()
    resnet34_label_predictor.cuda()
    class_logits = resnet34_label_predictor(
        resnet34_feature_extractor(test_data))
    soft_logits2 = nn.functional.softmax(class_logits, dim=1)
    resnet34_feature_extractor.cpu()
    resnet34_label_predictor.cpu()
    # Ensemble Part3: vgg16 acc:0.76144
    # vgg16_512_2_feature_extractor.cuda()
    # vgg16_512_2_label_predictor.cuda()
    # class_logits = vgg16_512_2_label_predictor(
    #     vgg16_512_2_feature_extractor(test_data))
    # soft_logits3 = nn.functional.softmax(class_logits, dim=1)
    # vgg16_512_2_feature_extractor.cpu()
    # vgg16_512_2_label_predictor.cpu()

    x = torch.argmax(1*soft_logits1+soft_logits2,
                     dim=1).cpu().detach().numpy()
    result.append(x)


result = np.concatenate(result)

# Generate your submission
df = pd.DataFrame({'id': np.arange(0, len(result)), 'label': result})
df.to_csv(sys.argv[2], index=False)
