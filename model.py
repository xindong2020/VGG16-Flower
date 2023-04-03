import torch
import torch.nn as nn


class VGG16(nn.Module):
    def __init__(self, vgg_name, init_weights=False):
        super(VGG16, self).__init__()
        cfg = {
            "VGG16": [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            "VGG19": [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
        }
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 10)
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, images):
        features = self.features(images)
        outputs = self.classifier(features.view(images.shape[0], -1))
        return outputs

    def _make_layers(self, cfg, in_channels=3):
        layers = []
        for layer in cfg:
            if layer == 'M':
                layers.append(nn.MaxPool2d(2, 2))
            else:
                layers.append(nn.Conv2d(in_channels, layer, 3, 1, 1))
                layers.append(nn.BatchNorm2d(layer))
                layers.append(nn.ReLU(inplace=True))
                in_channels = layer
        # * 作用在形参上，代表这个位置接收任意多个非关键字参数，转化成元组方式
        # * 作用在实参上，代表的是将输入迭代器拆成一个个元素
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, 0, 0.01)
                nn.init.constant_(layer.bias, 0)


# # 测试
# images = torch.rand([64, 3, 224, 224])
# model = VGG16("VGG16")
# outputs = model(images)
# print(outputs.shape)
