import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

# 非线性激活函数（可以根据需要更改）
class Expression(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)

def square(x):
    return x ** 2

def safe_log(x):
    return torch.log(x + 1e-6)

class globalnetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 40, kernel_size=(1, 50), stride=(1, 1), padding=0)
        init.xavier_uniform_(self.conv1.weight, gain=1)
        init.constant_(self.conv1.bias, 0)

        self.conv2 = nn.Conv2d(40, 40, kernel_size=(20, 1), stride=(1, 1), padding=0)
        init.xavier_uniform_(self.conv2.weight, gain=1)
        init.constant_(self.conv2.bias, 0)

        self.conv_nonlin = Expression(square)

        self.bn2 = nn.BatchNorm2d(40, momentum=0.1, affine=True)
        init.constant_(self.bn2.weight, 1)
        init.constant_(self.bn2.bias, 0)

        self.pool_nonlin = Expression(safe_log)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn2(self.conv2(x))
        x = self.conv_nonlin(x)
        x = F.avg_pool2d(x, (1, 150), stride=(1, 90))
        x = F.avg_pool2d(x, (1, 3), stride=(1, 4))
        x = self.pool_nonlin(x)

        return x


class localnetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 40, kernel_size=(1, 26), padding=0, stride=1)
        init.xavier_uniform_(self.conv1.weight, gain=1)
        init.constant_(self.conv1.bias, 0)

        self.conv2 = nn.Conv2d(40, 40, kernel_size=(20, 1), padding=0, stride=1)
        init.xavier_uniform_(self.conv2.weight, gain=1)
        init.constant_(self.conv2.bias, 0)

        self.conv12_nonlin = Expression(square)

        self.bn2 = nn.BatchNorm2d(40, momentum=0.1, affine=True)
        init.constant_(self.bn2.weight, 1)
        init.constant_(self.bn2.bias, 0)

        self.conv3 = nn.Conv2d(40, 40, kernel_size=(1, 9), padding=0, stride=1)
        init.xavier_uniform_(self.conv3.weight, gain=1)
        init.constant_(self.conv3.bias, 0)

        self.conv3_nonlin = Expression(square)

        self.bn3 = nn.BatchNorm2d(40, momentum=0.1, affine=True)
        init.constant_(self.bn3.weight, 1)
        init.constant_(self.bn3.bias, 0)

        self.conv4 = nn.Conv2d(40, 40, kernel_size=(1, 5), padding=0, stride=1)
        init.xavier_uniform_(self.conv4.weight, gain=1)
        init.constant_(self.conv4.bias, 0)

        self.conv4_nonlin = Expression(square)

        self.bn4 = nn.BatchNorm2d(40, momentum=0.1, affine=True)
        init.constant_(self.bn4.weight, 1)
        init.constant_(self.bn4.bias, 0)

        self.pool_nonlin = Expression(safe_log)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn2(self.conv2(x))
        x = self.conv12_nonlin(x)
        x = F.avg_pool2d(x, (1, 6), stride=(1, 6))
        x = self.pool_nonlin(x)
        x = self.bn3(self.conv3(x))
        x = self.conv3_nonlin(x)
        x = F.avg_pool2d(x, (1, 3), stride=(1, 3))
        x = self.pool_nonlin(x)
        x = self.bn4(self.conv4(x))
        x = self.conv4_nonlin(x)
        x = F.avg_pool2d(x, (1, 3), stride=(1, 3))
        x = self.pool_nonlin(x)

        return x


def _squeeze_final_output(x):
    #print(x.shape)
    assert x.size()[3] == 1
    x = x[:, :, :, 0]
    if x.size()[2] == 1:
        x = x[:, :, 0]
    return x


class topnetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(40, 4, kernel_size=(1, 11), stride=(1, 1))
        init.xavier_uniform_(self.conv1.weight, gain=1)
        init.constant_(self.conv1.bias, 0)

        self.log_softmax = nn.LogSoftmax(dim=1)
        self.squeeze = Expression(_squeeze_final_output)

    def forward(self, x):
        x = self.conv1(x)
        x = self.log_softmax(x)
        x = self.squeeze(x)

        return x

