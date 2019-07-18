import torch
import torch.nn as nn
from torch.optim import SGD, Adam
import torch.nn.functional as F
import threading
import time

# 기본적인 네트워크 모델
class test_net(nn.Module):
    def __init__(self):
        super(test_net, self).__init__()                        # 기본적인 틀
        self.layer = nn.Linear(in_features=3, out_features=2)   # 사용하고 싶은 레이어를 선언
        self.layer2 = nn.Linear(in_features=2, out_features=1)  # 사용하고 싶은 레이어를 선언

        self.optimizer = Adam(self.parameters(), 0.01)
        self.loss = nn.MSELoss()

    def forward(self, x):
        x = F.relu(self.layer(x))
        x = F.relu(self.layer2(x))
        return x

class Sample_net(threading.Thread):
    def __init__(self, net):
        threading.Thread.__init__(self)
        self.net = net

    def train(self, x, y):
        self.net.optimizer.zero_grad()
        out = self.net(x)
        loss = self.net.loss(out, y)
        loss.backward()
        self.net.optimizer.step()

    def run(self):
        x = torch.rand(3)
        y = torch.rand(1)
        print(x, y)
        print('이전 가중치 \n', self.net.layer.weight)
        self.train(x, y)
        print('이후 가중치 \n',self.net.layer.weight)

if __name__ == '__main__':
    # =============================================================
    # 핵심부분
    # 결론적으로 class 에 사전에 정의된 network 를 불러오고 해당 class 를 멀티 쓰레드로 돌려도
    # 훈련되는 것에는 아무런 문제가 되지 않는다.
    # -------------------------------------------------------------
    net = test_net()
    Sample_net_1 = Sample_net(net=net)
    Sample_net_2 = Sample_net(net=net)
    # =============================================================

    print('Network----------------------\n{}\n{}'.format(Sample_net_1.net, Sample_net_2.net))
    print('Weight-----------------------\n{}\n{}'.format(Sample_net_1.net.layer.weight,
                                                         Sample_net_2.net.layer.weight,))
    print('Train------------------------')
    thread_process = [Sample_net_1, Sample_net_2]
    for th in thread_process:
        th.start()
        time.sleep(1)