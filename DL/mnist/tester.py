import torch
import DataLoader
import Net
import os
import torch.nn.functional as F
import numpy as np

def Test(Batch_Size=64):
    model = Net.Net()
    if os.path.exists("./DL/mnist/model/mnist_net.pth"): #读取训练的模型
        model.load_state_dict(torch.load("./DL/mnist/model/mnist_net.pth"))
    
    Test_Loader = DataLoader.dataLoader(Train=False,Batch_size=Batch_Size)  # 获取测试集
    model.eval()
    test_Loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in Test_Loader:
            output = model(data)
            test_Loss += F.nll_loss(output, target, reduction='sum').item()  # 损失相加
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_Loss /= len(Test_Loader.dataset)

    print('\nTest: 平均损失: {:.4f}, 准确率: {}/{} ({:.2f}%)\n'.format(
        test_Loss, correct, len(Test_Loader.dataset),
        100. * correct/len(Test_Loader.dataset)
    ))

# Test()
