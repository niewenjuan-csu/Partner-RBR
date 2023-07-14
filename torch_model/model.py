# from torchinfo import summary
from torchsummary import summary
import torch
from torch import nn
from torch_model.config import DefaultConfig
from torch_model.TextCNN import TextCNN
from torch_model.DenseCNN import DenseCNN
from torch_model.BiGRU import BiGRU

configs = DefaultConfig()



class Bind(nn.Module):
    def __init__(self, num_classes):
        super(Bind, self).__init__()
        global configs

        self.dropout_rate = configs.dropout_dense
        sequence_channel = configs.sequence_channle
        structure_channel = configs.structure_channel

        # TextCNN_1
        output_dim = sequence_channel*4 + structure_channel*4
        self.conv_layer = nn.Sequential()
        self.conv_layer.add_module('conv',
                                   TextCNN(configs.windows_size, sequence_channel))

        self.conv_layer_1 = nn.Sequential()
        self.conv_layer_1.add_module('conv_neibor',
                                     TextCNN(configs.neibor_num, structure_channel))

        # Dense layers  input_dim,1024,256,6
        self.dense_layer1 = nn.Sequential()
        # self.dense_layer1.add_module("Dense1",
        #                              nn.Linear(output_dim, 1024))
        self.dense_layer1.add_module("Dense1",
                                     nn.Linear(output_dim, 1024))
        self.dense_layer1.add_module("Relu1",
                                     nn.ReLU())
        self.dense_layer1.add_module("Dropout1",
                                     nn.Dropout(self.dropout_rate))

        self.dense_layer2 = nn.Sequential()
        self.dense_layer2.add_module("Dense2",
                                     nn.Linear(1024, 256))
        self.dense_layer2.add_module("Relu2",
                                     nn.ReLU())
        self.dense_layer2.add_module("Dropout2",
                                     nn.Dropout(self.dropout_rate))

        self.out_layer = nn.Sequential(nn.Linear(256, num_classes),
                                       nn.Softmax(dim=1))


    def forward(self, x, neibor_x):
        # TextCNN:
        fea = self.conv_layer(x)
        neibor_fea = self.conv_layer_1(neibor_x)

        fea = torch.cat((fea, neibor_fea), dim=-1)

        output = self.dense_layer1(fea)
        output = self.dense_layer2(output)
        output = self.out_layer(output)

        return output, fea


if __name__ == '__main__':
    model = Bind(num_classes=6)
    summary(model, input_size=(1, 11, 1169))
    # summary(model, input_size=[(1, 11, 1169),(1,9,1169)])

