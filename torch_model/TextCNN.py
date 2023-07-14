import math
from torchsummary import summary
# from torchinfo import summary

import torch
from torch import nn
from torch_model.config import DefaultConfig

configs = DefaultConfig()

# TextCNN
# Input: [batch, C, H, W]
class TextCNN(nn.Module):
    def __init__(self, feature_L, num_filter):
        super(TextCNN, self).__init__()

        in_channle = 1
        self.kernel = configs.kernel
        kernel_width = configs.fea_dim

        padding1 = (self.kernel[0] - 1) // 2
        padding2 = (self.kernel[1] - 1) // 2
        padding3 = (self.kernel[2] - 1) // 2
        padding4 = (self.kernel[3] - 1) // 2

        self.convlayer1 = nn.Sequential(nn.Conv2d(in_channels=in_channle, out_channels=num_filter,
                                                  kernel_size=(self.kernel[0], kernel_width),
                                                  padding=(padding1, 0), bias=False),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=(feature_L, 1), stride=1))
        self.convlayer2 = nn.Sequential(nn.Conv2d(in_channels=in_channle, out_channels=num_filter,
                                                  kernel_size=(self.kernel[1], kernel_width),
                                                  padding=(padding2, 0), bias=False),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=(feature_L, 1), stride=1))
        self.convlayer3 = nn.Sequential(nn.Conv2d(in_channels=in_channle, out_channels=num_filter,
                                                  kernel_size=(self.kernel[2], kernel_width),
                                                  padding=(padding3, 0), bias=False),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=(feature_L, 1), stride=1))
        self.convlayer4 = nn.Sequential(nn.Conv2d(in_channels=in_channle, out_channels=num_filter,
                                                  kernel_size=(self.kernel[3], kernel_width),
                                                  padding=(padding4, 0), bias=False),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=(feature_L, 1), stride=1))

    def forward(self, x):
        x1 = self.convlayer1(x)
        x2 = self.convlayer2(x)
        x3 = self.convlayer3(x)
        x4 = self.convlayer4(x)
        x = torch.cat((x1, x2, x3, x4), dim=1)

        shape = x.data.shape
        x = x.view(shape[0], shape[1]*shape[2]*shape[3])

        return x


if __name__ == '__main__':
    model = TextCNN(9, 32)
    summary(model, input_size=(1, 9, 1169))


