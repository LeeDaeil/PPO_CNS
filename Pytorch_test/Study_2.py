import torch
import torch.nn.functional as F
from torch.autograd import Variable

# 입력 값과 필터의 값을 작성
input_value = torch.ones(1, 1, 3, 3)
filter_value = torch.ones(1, 1, 3, 3)

input_value = Variable(input_value, requires_grad=True)
print(input_value)

filter_value = Variable(filter_value)
print(filter_value)

# conv2d에 계산
out = F.conv2d(input_value, filter_value)
print(out)

out.backward()
print(out.grad_fn)
print(input_value.grad)

import torch
import torch.nn as nn

func = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3)
print(func)
