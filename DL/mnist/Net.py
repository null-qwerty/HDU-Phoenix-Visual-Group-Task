import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)  # 卷积层1
        # nn.Conv2d(输入通道数, 输出通道数, 卷积核尺寸, 步长(默认1))
        self.conv2 = nn.Conv2d(32, 64, 3, 1)  # 卷积层2
        self.dropout1 = nn.Dropout(0.25)  # dropout减少过拟合
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)  # in:1*28*28     out:32*26*26
        x = F.relu(x)      
        x = self.conv2(x)  # in:32*26*26    out:64*24*24
        x = F.relu(x)
        x = F.max_pool2d(x, 2)  # 池化 in:64*24*24   out: 64*12*12
        x = self.dropout1(x)
        x = torch.flatten(x, 1)  # 扁平化 in: 64*12*12  out: 9216*1
        x = self.fc1(x) # in:9216*1     out: 128*1
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x) # in:128*1      out:10*1
        output = F.log_softmax(x, dim=1)  #输出
        return output
