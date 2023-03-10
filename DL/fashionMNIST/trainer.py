import torch
from torch.optim import Adam
import DataLoader
import os
import torch.nn.functional as F
from tensorboardX import SummaryWriter


def Train(epoch, Model, Device, Batch_Size=64, Net_Name="", Img_Size=28):
    # 模型及折线图保存路径
    path = "./DL/fashionMNIST/model/"+Net_Name
    logger = SummaryWriter(log_dir="./DL/fashionMNIST/log/"+Net_Name)

    model = Model  # 神经网络模型
    optimizer = Adam(model.parameters(), lr=0.001)  # 优化器
    if os.path.exists(path+"/fashion_mnist_net.pth"):  # 读取训练的模型
        model.load_state_dict(torch.load(path+"/fashion_mnist_net.pth"))
        optimizer.load_state_dict(torch.load(path+"/fashion_mnist_optimizer.pth"))

    Train_Loader = DataLoader.dataLoader(Train=True, Batch_size=Batch_Size, Size=Img_Size)  # 获取测试集
    model.train()
    for idx, (data, target) in enumerate(Train_Loader):
        # 变量初始化
        data, target = data.to(Device), target.to(Device)
        optimizer.zero_grad()  # 优化器梯度设置为0
        # 训练
        output = model(data)  # 数据读入网络
        loss = F.nll_loss(output, target)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 优化器权重更新
        # 统计准确率
        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        acc = correct/data.shape[0]
        # 输出
        if idx % 10 == 0:
            print('{} Train Epoch {}: [{}/{} ({:.0f}%)]\tLoss:{:.6f}'.format(
                Net_Name, epoch, idx*len(data), len(Train_Loader.dataset),
                100. * idx/len(Train_Loader), loss.item()
            ))
            # 绘制折线图
            logger.add_scalar("train loss", loss.item(), global_step=idx)
            logger.add_scalar("train accuary", acc, global_step=idx)
    # 保存模型参数，state_dict用来获取数据，save用来保存数据
    torch.save(model.state_dict(), path+"/fashion_mnist_net.pth")
    torch.save(optimizer.state_dict(), path+"/fashion_mnist_optimizer.pth")  # 保存优化器


# for i in range(3):
#     Train(i)
