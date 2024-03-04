import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
from torchvision.transforms import transforms


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=100):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.layer5 = self.make_layer(ResidualBlock, 1024, 2, stride=2)
        self.layer6 = self.make_layer(ResidualBlock, 2048, 2, stride=2)
        self.fc = nn.Linear(2048, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = F.avg_pool2d(out,2)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def ResNet18():
    return ResNet(ResidualBlock)

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

class MyDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.img_list = os.listdir(self.path)
        self.img_path = []
        for i in self.img_list:
            self.img_path.append('test//' + i)

    def __getitem__(self,idx):
        img_name = self.img_path[idx]
        img = Image.open(img_name).convert('RGB')
        img = transform_test(img)
        return img

    def __len__(self):
        return len(self.img_list)



testset = MyDataset('test')
testloader = DataLoader(testset, batch_size=128, shuffle=False)

#加载已有模型
model = torch.load('net.pth')
txt_obj = open('test.txt', 'a', encoding='UTF-8')
num = 0

#测试模型
with torch.no_grad():
    for i, data in enumerate(testloader, 0):
        # 准备数据
        length = len(testloader)
        inputs = data
        inputs = inputs.to('cuda')
        model.eval()
        outputs = model(inputs).argmax(1)
        for output in outputs:
            output =int(output)
            txt_obj.write('test/' + str(num) + '.jpg' + ' ' + str(output) + '\n')
            num += 1
        print(outputs)