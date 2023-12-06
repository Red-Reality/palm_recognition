import random
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models
from torch.utils.data import Dataset, DataLoader
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os


class CustomDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.images1, self.images2, self.labels = self.load_images()

    def load_images(self):
        images1 = []
        images2 = []
        labels = []

        # 遍历目录中的所有文件
        tmplist = []
        for filename in os.listdir(self.directory):
            tmplist.append(filename)
        for i in range(10000):
            filename1 = tmplist[random.randint(0, len(tmplist) - 1)]
            filename2 = tmplist[random.randint(0, len(tmplist) - 1)]

            images1.append(os.path.join(self.directory, filename1))
            images2.append(os.path.join(self.directory, filename2))

            labels.append(filename1.split('_')[0] == filename2.split('_')[0])

        return images1, images2, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_path1 = self.images1[idx]
        image_path2 = self.images2[idx]
        image1 = Image.open(image_path1)
        image2 = Image.open(image_path2)

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        label = self.labels[idx]
        return image1, image2, label


# 定义 Siamese 网络结构
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # 使用 MobileNet 作为特征提取器
        self.mobilenet = models.mobilenet_v2(pretrained=True)
        self.mobilenet.classifier = nn.Identity()  # 移除分类层

        # 添加额外的分类层
        self.classifier = nn.Sequential(
            nn.Linear(self.mobilenet.last_channel * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, input1, input2):
        # 通过相同的网络提取特征
        output1 = self.mobilenet(input1)
        output2 = self.mobilenet(input2)
        # 特征组合
        combined = torch.cat((output1, output2), dim=1)

        # 分类
        output = self.classifier(combined)
        return output


# 对比损失函数
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive


if __name__ == '__main__':
    # 实例化模型
    model = SiameseNetwork()

    # 加载状态字典
    model.load_state_dict(torch.load('./curr_0(1).pth', map_location=torch.device('cpu')))

    # 设置为评估模式
    model.eval()
    transform = transforms.Compose([
        transforms.Resize([224, 224]),  # 根据需要调整大小
        transforms.ToTensor(),
        # 根据需要添加其他变换
    ])
    times = 5000
    acc = 0
    imgroot = r'C:\Users\leon\Desktop\computer_vision\final_project\MPDv2\generations\2-ROI verification\ROI\ROI'
    tmpfilename = []
    for file in os.listdir(imgroot):
        tmpfilename.append(file)
    for i in range(times):
        img1root = tmpfilename[random.randint(0, len(tmpfilename) - 1)]
        img2root = tmpfilename[random.randint(0, len(tmpfilename) - 1)]
        image1 = Image.open(imgroot + '/' + img1root)
        image2 = Image.open(imgroot + '/' + img2root)

        # 应用转换
        image1 = transform(image1)
        image2 = transform(image2)

        # 添加批次维度
        image1 = image1.unsqueeze(0)
        image2 = image2.unsqueeze(0)
        # 假设使用的是CPU
        output = model(image1, image2)
        similarity = output[0][0]
        if similarity>0.6:
            similarity=1
        else:
            similarity=0
        (labelnum1, labelr1) = (img1root.split('_')[0], img1root.split('_')[3])
        (labelnum2, labelr2) = (img2root.split('_')[0], img2root.split('_')[3])
        label = (labelnum1, labelr1) ==(labelnum2, labelr2)
        if similarity==label:
            acc+=1
        if i%100==0:
            print(f"current times is {i}")
    print(f"acc is {acc/times}")
