from trainer import Train
from tester import Test

Test(Batch_Size=1024)
for i in range(3):
    Train(epoch=i+1,Batch_Size=128)
Test(Batch_Size=1024)