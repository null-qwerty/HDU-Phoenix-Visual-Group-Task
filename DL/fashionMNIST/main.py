from trainer import Train
from tester import Test
import torch
import CNN

# 使用fashion-mnist, 输入为28*28灰度图
# cuda
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# LeNet
model = CNN.LeNet()
model = model.to(device)
Test(Model=model, Device=device, Batch_Size=1024, Net_Name='LeNet')
Train(epoch=0, Model=model, Device=device, Batch_Size=128, Net_Name='LeNet')
Test(Model=model, Device=device, Batch_Size=1024, Net_Name='LeNet')

# AlexNet
# AlexNet输入尺寸224*224, 需要扩大尺寸
model = CNN.AlexNet()
model = model.to(device)
Test(Model=model, Device=device, Batch_Size=1024, Net_Name='AlexNet', Img_Size=224)
Train(epoch=0, Model=model, Device=device, Batch_Size=128, Net_Name='AlexNet', Img_Size=224)
Test(Model=model, Device=device, Batch_Size=1024, Net_Name='AlexNet', Img_Size=224)

# VGG_16
# VGG_16输入尺寸224*224, 需要扩大尺寸
model = CNN.VGG_16()
model = model.to(device)
Test(Model=model, Device=device, Batch_Size=1024, Net_Name='VGG_16', Img_Size=224)
Train(epoch=0, Model=model, Device=device, Batch_Size=128, Net_Name='VGG_16', Img_Size=224)
Test(Model=model, Device=device, Batch_Size=1024, Net_Name='VGG_16', Img_Size=224)
