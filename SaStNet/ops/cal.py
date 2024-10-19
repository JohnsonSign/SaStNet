##测试计算feature map


import torch
import time

x = torch.zeros((8,64,1000,1000))
y = torch.ones((8,64,1000,1000))

print(x.size())
print(y.size())

start = time.time()
for i in range(1000):
    x1 = torch.mul(x,x)
end = time.time()
print('x1 %s seconds'%(end - start))

start1 = time.time()
for i in range(1000):
    y1 = torch.mul(y,y)
end1 = time.time()
print('y1 %s seconds'%(end1 - start1))
