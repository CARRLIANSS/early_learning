import torch
import torch.nn as nn
from models.deepda_losses import DeepDA_Losses
import models.backbones

class DeepDANet(nn.Module):
    def __init__(self, num_class, base_net='resnet50', loss_type='mmd', use_bottleneck=True, bottleneck_width=256, **kwargs):
        super(DeepDANet, self).__init__()
        self.num_class = num_class
        self.use_bottleneck = use_bottleneck # 此处的bottleneck使用的是全连接
        self.loss_type = loss_type

        # backbone
        self.backbone = models.backbones.get_backbone(base_net)

        # bottleneck
        if self.use_bottleneck:
            bottleneck_list = [
                nn.Linear(self.backbone.output_num(), bottleneck_width),
                nn.ReLU()
            ]
            self.bottleneck = nn.Sequential(*bottleneck_list)
            feature_dim = bottleneck_width
        else:
            feature_dim = self.backbone.output_num()

        # classifier
        self.classifier_layer = nn.Linear(feature_dim, num_class)

        # loss
        # Domain Adaptation的精髓在于loss function，所以将loss放入网络中
        deepda_loss_args = {
            "loss_type": self.loss_type,
            "num_class": num_class
        }
        self.deepda_loss = DeepDA_Losses(**deepda_loss_args)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, source, source_label, target):
        source = self.backbone(source)
        target = self.backbone(target)

        if self.use_bottleneck:
            source = self.bottleneck(source)
            target = self.bottleneck(target)

        source_clf = self.classifier_layer(source)

        kwargs = {}
        if self.loss_type == 'bnm':
            tar_clf = self.classifier_layer(target)
            target = nn.Softmax(dim=1)(tar_clf)

        clf_loss = self.criterion(source_clf, source_label)
        deepda_loss = self.deepda_loss(source, target, **kwargs)

        return clf_loss, deepda_loss

    def get_parameters(self, initial_lr=1.0):
        params = [
            {'params': self.backbone.parameters(), 'lr': 0.1 * initial_lr},
            {'params': self.classifier_layer.parameters(), 'lr': 1.0 * initial_lr},
        ]
        if self.use_bottleneck:
            params.append(
                {'params': self.bottleneck.parameters(), 'lr': 1.0 * initial_lr}
            )
        # Loss-dependent
        if self.deepda_loss == "adv":
            params.append(
                {'params': self.deepda_loss.loss_func.domain_classifier.parameters(), 'lr': 1.0 * initial_lr}
            )
        elif self.deepda_loss == "daan":
            params.append(
                {'params': self.deepda_loss.loss_func.domain_classifier.parameters(), 'lr': 1.0 * initial_lr}
            )
            params.append(
                {'params': self.deepda_loss.loss_func.local_classifiers.parameters(), 'lr': 1.0 * initial_lr}
            )
        return params

    def predict(self, x):
        x = self.backbone(x)
        if self.use_bottleneck:
            x = self.bottleneck(x)
        x = self.classifier_layer(x)
        return x