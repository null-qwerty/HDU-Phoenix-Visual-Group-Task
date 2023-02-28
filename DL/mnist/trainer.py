import torch
from torch.optim import Adam
import DataLoader
import Net
import os
import torch.nn.functional as F

def Train(epoch, Batch_Size=64):
    model = Net.Net()  # 神经网络模型
    optimizer = Adam(model.parameters(), lr=0.001)  # 优化器
    if os.path.exists("./DL/mnist/model/mnist_net.pth"):  # 读取训练的模型
        model.load_state_dict(torch.load("./DL/mnist/model/mnist_net.pth"))
        optimizer.load_state_dict(torch.load(
            "./DL/mnist/model/mnist_optimizer.pth"))
    
    Train_Loader = DataLoader.dataLoader(Train=True,Batch_size=Batch_Size)  # 获取测试集
    model.train()
    for idx, (data, target) in enumerate(Train_Loader):
        optimizer.zero_grad()  # 优化器梯度设置为0
        output = model(data)  # 数据读入网络
        loss = F.nll_loss(output, target)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 优化器权重更新
        if idx % 10 == 0:
            print('Train Epoch {}: [{}/{} ({:.0f}%)]\tLoss:{:.6f}'.format(
                epoch, idx*len(data), len(Train_Loader.dataset),
                100. * idx/len(Train_Loader), loss.item()
            ))
    # 保存模型参数，state_dict用来获取数据，save用来保存数据
    torch.save(model.state_dict(),
               "./DL/mnist/model/mnist_net.pth")
    torch.save(optimizer.state_dict(),
               "./DL/mnist/model/mnist_optimizer.pth")  # 保存优化器


# for i in range(3):
#     Train(i)
