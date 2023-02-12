""" 
    摘自网络，学习完毕后重新自写
"""

from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import os
import torch
import numpy as np

BATCH_SIZE = 128
TEST_BATCH_SIZE = 1000
# 准备数据集

def get_dataloader(train, batch_size=BATCH_SIZE):
    transform_fn = Compose([
        ToTensor(),
        Normalize(mean=(0.1307,), std=(0.3081,))
    ])  # mean和std的形状与通道数相同

    dataset = MNIST(root='./DL/mnist/Dataset', train=train, transform=transform_fn)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader


class MnistModel(nn.Module):
    def __init__(self):
        super(MnistModel, self).__init__()  # 继承
        self.fc1 = nn.Linear(1*28*28, 28)  # 参数是input和output的feature
        self.fc2 = nn.Linear(28, 10)

    def forward(self, input):
        # 1.进行形状的修改
        x = input.view([-1, 1*28*28])  # -1表示根据形状自动调整，也可以改为input.size(0)
        # 2.进行全连接的操作
        x = self.fc1(x)
        # 3.激活函数的处理
        x = F.relu(x)  # 形状没有变化
        # 4.输出层
        out = self.fc2(x)
        return F.log_softmax(out, dim=-1)


model = MnistModel()
optimizer = Adam(model.parameters(), lr=0.001)
if os.path.exists("./DL/mnist/model/mnist_net.pth"): #读取训练的模型
        model.load_state_dict(torch.load("./DL/mnist/model/mnist_net.pth"))
        optimizer.load_state_dict(torch.load("./DL/mnist/model/mnist_optimizer.pth"))


def train(epoch):  # epoch表示几轮
    data_loader = get_dataloader(True)  # 获取数据加载器
    # idx表示data_loader中的第几个数据，元组是data_loader的数据
    for idx, (input, target) in enumerate(data_loader):
        optimizer.zero_grad()  # 将梯度置0
        output = model(input)  # 调用模型，得到预测值
        loss = F.nll_loss(output, target)  # 调用损失函数，得到损失,是一个tensor
        loss.backward()  # 反向传播
        optimizer.step()  # 梯度的更新
        if idx % 10 == 0:
            print(epoch, idx, loss.item())
    # for是每一轮中的数据进行遍历
    torch.save(model.state_dict(),"./DL/mnist/model/mnist_net.pth")#保存模型参数，state_dict用来获取数据，save用来保存数据
    torch.save(optimizer.state_dict(),"./DL/mnist/model/mnist_optimizer.pth")#保存优化器

def test():
    loss_list = []
    acc_list = []
    test_dataloader = get_dataloader(
        train=False, batch_size=TEST_BATCH_SIZE)  # 获取测试集
    for idx, (input, target) in enumerate(test_dataloader):
        with torch.no_grad():  # 不计算梯度
            output = model(input)
            cur_loss = F.nll_loss(output, target)
            loss_list.append(cur_loss)
            # 计算准确率，output大小[batch_size,10] target[batch_size] batch_size是多少组数据，10列是每个数字概率
            pred = output.max(dim=-1)[-1]  # 获取最大值位置
            cur_acc = pred.eq(target).float().mean()
            acc_list.append(cur_acc)
    print("平均准确率：", np.mean(acc_list), "平均损失：", np.mean(loss_list))


test()
for i in range(3):  # 训练三轮
    train(i)
test()