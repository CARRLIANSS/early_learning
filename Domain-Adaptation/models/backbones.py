import torch.nn as nn
from torchvision import models

resnet_dict = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
    "resnet101": models.resnet101,
    "resnet152": models.resnet152
}

def get_backbone(name):
    if "resnet" in name.lower():
        return ResnetBackbone(name)
    elif "alexnet" == name.lower():
        return AlexnetBackbone(name)
    
    
class ResnetBackbone(nn.Module):
    
    def __init__(self, network_type):
        super(ResnetBackbone, self).__init__()
        resnet = resnet_dict[network_type](pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        self._feature_dim = resnet.fc.in_features
        del resnet

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1) # 将bs维度固定
        return x

    def output_num(self):
        return self._feature_dim


class AlexnetBackbone(nn.Module):

    def __init__(self):
        super(AlexnetBackbone, self).__init__()
        Alexnet = models.alexnet(pretrained=True)
        self.features = Alexnet.features
        self.classifier = nn.Sequential()
        for i in range(6):
            self.classifier.add_module("classifier"+str(i), Alexnet.classifier[i])
        self._feature_dim = Alexnet.classifier[6].in_features

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256*6*6)
        x = self.classifier(x)
        return x

    def output_num(self):
        return self._feature_dim

