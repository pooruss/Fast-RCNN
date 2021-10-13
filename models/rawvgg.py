import paddle.nn as nn
import paddle
import numpy
import numpy as np


class VGG(nn.Layer):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        # define an empty for Conv_ReLU_MaxPool
        net = []

        # block 1
        net.append(nn.Conv2D(in_channels=3, out_channels=64, padding=1, kernel_size=3, stride=1))
        net.append(nn.BatchNorm2D(64))
        net.append(nn.ReLU())
        net.append(nn.Conv2D(in_channels=64, out_channels=64, padding=1, kernel_size=3, stride=1))
        net.append(nn.BatchNorm2D(64))
        net.append(nn.ReLU())
        net.append(nn.MaxPool2D(kernel_size=2, stride=2))

        # block 2
        net.append(nn.Conv2D(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1))
        net.append(nn.BatchNorm2D(128))
        net.append(nn.ReLU())
        net.append(nn.Conv2D(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1))
        net.append(nn.BatchNorm2D(128))
        net.append(nn.ReLU())
        net.append(nn.MaxPool2D(kernel_size=2, stride=2))

        # block 3
        net.append(nn.Conv2D(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1))
        net.append(nn.BatchNorm2D(256))
        net.append(nn.ReLU())
        net.append(nn.Conv2D(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1))
        net.append(nn.BatchNorm2D(256))
        net.append(nn.ReLU())
        net.append(nn.Conv2D(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1))
        net.append(nn.BatchNorm2D(256))
        net.append(nn.ReLU())
        #net.append(nn.MaxPool2D(kernel_size=2, stride=2))

        # block 4
        net.append(nn.Conv2D(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=1))
        net.append(nn.BatchNorm2D(512))
        net.append(nn.ReLU())
        net.append(nn.Conv2D(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1))
        net.append(nn.BatchNorm2D(512))
        net.append(nn.ReLU())
        net.append(nn.Conv2D(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1))
        net.append(nn.BatchNorm2D(512))
        net.append(nn.ReLU())
        #net.append(nn.MaxPool2D(kernel_size=2, stride=2))

        # block 5
        net.append(nn.Conv2D(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1))
        net.append(nn.BatchNorm2D(512))
        net.append(nn.ReLU())
        net.append(nn.Conv2D(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1))
        net.append(nn.BatchNorm2D(512))
        net.append(nn.ReLU())
        net.append(nn.Conv2D(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1))
        net.append(nn.BatchNorm2D(512))
        net.append(nn.ReLU())
        #net.append(nn.MaxPool2D(kernel_size=2, stride=2))

        # add net into class property
        self.extract_feature = nn.Sequential(*net)

    # fullvggnet
    # def forward(self, x):
    #     feature = self.extract_feature(x)
    #     print(feature.shape)
    #     feature = paddle.reshape(feature, [0, -1])
    #     torch:feature = feature.view(x.size(0), -1)
    #     classify_result = self.classifier(feature)
    #     return classify_result

    # rawnet
    def forward(self, x):
        #print(x.dtype)
        feature = self.extract_feature(x)
        return feature

# test
# if __name__ == "__main__":
#     x = paddle.rand(shape=(8, 3, 224, 224))
#     vgg = VGG(num_classes=80)
#     out = vgg(x)
    #print(out.shape)   (8,80)