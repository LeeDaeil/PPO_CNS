import torch
import torch.nn as nn
from torch.optim import SGD
import torch.nn.functional as F

# 기본적인 네트워크 모델
class test_net(nn.Module):
    def __init__(self):
        super(test_net, self).__init__()                        # 기본적인 틀
        self.layer = nn.Linear(in_features=3, out_features=2)   # 사용하고 싶은 레이어를 선언
        self.layer2 = nn.Linear(in_features=2, out_features=1)  # 사용하고 싶은 레이어를 선언

    def forward(self, x):
        x = F.relu(self.layer(x))
        x = F.relu(self.layer2(x))
        # x = self.layer(x)
        # x = self.layer2(x)
        return x

a = torch.rand(1)
b = torch.rand(3)
print(a, b)

# 네트워크 생성
net = test_net()
# 네트워크에 입력
print(net(b))

# 옵티마이저 생성
opt = SGD(net.parameters(), lr=0.001)
# loss function 선언
loss_func = nn.MSELoss()
print('----')
for _ in range(0, 10):
    out = net(b)
    loss = loss_func(out, a)
    loss.backward()
    opt.step()

    print(net(b))

