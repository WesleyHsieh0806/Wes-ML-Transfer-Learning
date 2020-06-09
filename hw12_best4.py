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

#### 一些細節

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


class FeatureExtractor(nn.Module):

    def __init__(self):
        super(FeatureExtractor, self).__init__()
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


class LabelPredictor(nn.Module):

    def __init__(self):
        super(LabelPredictor, self).__init__()

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


class DomainClassifier(nn.Module):

    def __init__(self):
        super(DomainClassifier, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 1),
        )

    def forward(self, h):
        y = self.layer(h)
        return y


"""# Pre-processing

這裡我們選用Adam來當Optimizer。
"""

feature_extractor = FeatureExtractor().cuda()
# feature_extractor.load_state_dict(torch.load(
#     './result/vgg16_512/vgg16_extractor_model.bin'))
label_predictor = LabelPredictor().cuda()
# label_predictor.load_state_dict(torch.load(
# './result/vgg16_512/vgg16_predictor_model.bin'))
domain_classifier = DomainClassifier().cuda()
# domain_classifier.load_state_dict(torch.load(
# './result/vgg16_512/vgg16_domain_model.bin'))

class_criterion = nn.CrossEntropyLoss()
domain_criterion = nn.BCEWithLogitsLoss()
# BCE : -1/n  yi*ln(xi)+(1-yi)*ln(1-xi)
# BSE with logits loss則是先把x做sigmoid
# logits: 不是數學統計上常用的logits 這裡的logits是指自動將input normalize到0~1
optimizer_F = optim.Adam(
    feature_extractor.parameters(), lr=0.001)
# optimizer_F = optim.SGD(
# feature_extractor.parameters(), lr=0.0005, momentum=0.9)
# C 代表classification
optimizer_C = optim.Adam(label_predictor.parameters())
# optimizer_C = optim.SGD(label_predictor.parameters(), lr=0.001, momentum=0.9)
optimizer_D = optim.Adam(
    domain_classifier.parameters(), lr=0.001)
# optimizer_D = optim.SGD(
# domain_classifier.parameters(), lr=0.001, momentum=0.9)
"""# Start Training


## 如何實作DaNN?

理論上，在原始paper中是加上Gradient Reversal Layer，並將Feature Extractor / Label Predictor / Domain Classifier 一起train，但其實我們也可以交換的train Domain Classfier & Feature Extractor(就像在train GAN的Generator & Discriminator一樣)，這也是可行的。

在code實現中，我們採取後者的方式，畢竟大家上個作業就是GAN，應該會比較熟悉:)。

## 小提醒
* 原文中的lambda(控制Domain Adversarial Loss的係數)是有Adaptive的版本，如果有興趣可以參考[原文](https://arxiv.org/pdf/1505.07818.pdf)。以下為了方便固定設置0.1。
* 因為我們完全沒有target的label，所以結果如何，只好丟kaggle看看囉:)?
"""


def train_epoch(source_dataloader, target_dataloader, lamb):
    '''
      Args:
        source_dataloader: source data的dataloader
        target_dataloader: target data的dataloader
        lamb: 調控adversarial的loss係數。
    '''

    # D loss: Domain Classifier的loss
    # F loss: Feature Extrator & Label Predictor的loss
    # total_hit: 計算目前對了幾筆 total_num: 目前經過了幾筆
    running_D_loss, running_F_loss = 0.0, 0.0
    total_hit, total_num = 0.0, 0.0

    for i, ((source_data, source_label), (target_data, _)) in enumerate(zip(source_dataloader, target_dataloader)):

        source_data = source_data.cuda()
        source_label = source_label.cuda()
        target_data = target_data.cuda()

        # 我們把source data和target data混在一起，否則batch_norm可能會算錯 (兩邊的data的mean/var不太一樣)
        mixed_data = torch.cat([source_data, target_data], dim=0)
        domain_label = torch.zeros(
            [source_data.shape[0] + target_data.shape[0], 1]).cuda()
        # 設定source data的label為1
        domain_label[:source_data.shape[0]] = 1

        # Step 1 : 訓練Domain Classifier
        feature = feature_extractor(mixed_data)
        # 因為我們在Step 1不需要訓練Feature Extractor，所以把feature detach避免loss backprop上去。
        domain_logits = domain_classifier(feature.detach())
        loss = domain_criterion(domain_logits, domain_label)
        running_D_loss += loss.item()
        loss.backward()
        optimizer_D.step()

        # Step 2 : 訓練Feature Extractor和Domain Classifier
        class_logits = label_predictor(feature[:source_data.shape[0]])
        domain_logits = domain_classifier(feature)
        # loss為原本的class CE - lamb * domain BCE，相減的原因同GAN中的Discriminator中的G loss。
        loss = class_criterion(class_logits, source_label) - \
            lamb * domain_criterion(domain_logits, domain_label)
        running_F_loss += loss.item()
        loss.backward()
        optimizer_F.step()
        optimizer_C.step()

        optimizer_D.zero_grad()
        optimizer_F.zero_grad()
        optimizer_C.zero_grad()

        total_hit += torch.sum(torch.argmax(class_logits,
                                            dim=1) == source_label).item()
        total_num += source_data.shape[0]
        print("[{}/{}]".format(i, len(source_dataloader)), end='\r')

    return running_D_loss / (i+1), running_F_loss / (i+1), total_hit / total_num


# 訓練200 epochs
best_loss = np.inf
best_acc = 0.
feature_extractor_loss = []
domain_loss = []
acc = []
num_epoch = 1500
# gamma用在計算lamb(Adaptation parameter)
gamma = 10
print("Total # of parameters of Feature Extractor:{}".format(
    sum(p.numel() for p in feature_extractor.parameters())))
print("Total # of parameters of label Predictor:{}".format(
    sum(p.numel() for p in label_predictor.parameters())))
print("Total # of parameters of Domain Classifier:{}".format(
    sum(p.numel() for p in domain_classifier.parameters())))
for epoch in range(num_epoch):
    start = time.time()
    # p用在adaptation parameter
    p = (epoch)*1/(num_epoch-1)
    lamb = (2/(1+math.exp(-gamma*p)) - 1)
    train_D_loss, train_F_loss, train_acc = train_epoch(
        source_dataloader, target_dataloader, lamb=lamb)
    if train_F_loss < best_loss and best_acc < train_acc:
        best_loss = train_F_loss
        best_acc = train_acc
        print("Save Model with domain loss {:.6f} acc {:.6f} and loss {:.6f}...".format(
            train_D_loss, train_acc, best_loss))
        torch.save(feature_extractor.state_dict(),
                   f'./result/vgg16_512/vgg16_extractor_model.bin')
        torch.save(label_predictor.state_dict(),
                   f'./result/vgg16_512/vgg16_predictor_model.bin')
        torch.save(domain_classifier.state_dict(),
                   f'./result/vgg16_512/vgg16_domain_model.bin')
    feature_extractor_loss.append(train_F_loss)
    domain_loss.append(train_D_loss)
    acc.append(train_acc)
    print('epoch {:>3d}: {:.3f} Secs | train D loss: {:6.4f}, train F loss: {:6.4f}, acc {:6.4f}'.format(
        epoch, time.time()-start, train_D_loss, train_F_loss, train_acc))
# record the training process(loss)
print("The smallest loss is at epoch:{}".format(
    np.argsort(np.array(feature_extractor_loss))[0]+1))

plt.plot(feature_extractor_loss, 'b-')
plt.title('feature extractor loss')
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.savefig('./result/vgg16_512/feature_extractor_loss')
plt.close()
df = pd.DataFrame(data=np.array(feature_extractor_loss).reshape(
    1, len(feature_extractor_loss)), columns=['epoch'+str(i) for i in range(1, num_epoch+1)], index=['FE loss'])
df.to_csv('./result/feature_extractor_loss.csv')
plt.plot(domain_loss, 'b-')
plt.title('Domain discriminator loss')
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.savefig('./result/vgg16_512/Domain_discriminator_loss')
plt.close()
df = pd.DataFrame(data=np.array(domain_loss).reshape(
    1, len(domain_loss)), columns=['epoch'+str(i) for i in range(1, num_epoch+1)], index=['Domain loss'])
df.to_csv('./result/vgg16_512/feature_extractor_loss.csv')

plt.plot(acc, 'b-')
plt.title('Accuracy')
plt.xlabel('epoch')
plt.ylabel('Accuracy')
plt.savefig('./result/vgg16_512/Accuracy')
plt.close()
"""# Inference

就跟前幾次作業一樣。這裡我使用pd來生產csv，因為看起來比較潮(?)

此外，200 epochs的Accuracy可能會不太穩定，可以多丟幾次或train久一點。
"""

result = []
feature_extractor.load_state_dict(torch.load(
    './result/vgg16_512/vgg16_extractor_model.bin'))
label_predictor.load_state_dict(torch.load(
    './result/vgg16_512/vgg16_predictor_model.bin'))
feature_extractor.eval()
label_predictor.eval()
for i, (test_data, _) in enumerate(test_dataloader):
    test_data = test_data.cuda()

    class_logits = label_predictor(feature_extractor(test_data))

    x = torch.argmax(class_logits, dim=1).cpu().detach().numpy()
    result.append(x)

result = np.concatenate(result)

# Generate your submission
df = pd.DataFrame({'id': np.arange(0, len(result)), 'label': result})
df.to_csv(sys.argv[2], index=False)
