import torch

# 텐서 만들기
x = torch.Tensor(3)
print(x)

# 랜덤 텐서 만들기
print(torch.rand(3, 3))     # random
print(torch.randn(3, 3))    # normal distribution

# 넘파이 행렬에서 텐서 만들기
import numpy as np
a = np.array([1, 2, 3, 4])
print(torch.Tensor(a))

# 텐서에서 넘파이로
a = torch.Tensor(3, 3)
b = a.numpy()
print(b, type(b))

# 텐서의 형태 변환
a = torch.Tensor(3, 3)
a = a.view(1, 1, 3, 3)
print(a)

# 텐서 합치기
a = torch.Tensor(np.array([[1, 1, 1, 1], [2, 2, 2, 2]]))
b = torch.Tensor(np.array([[3, 3, 3, 3], [4, 4, 4, 4]]))
print(a, b)
print(torch.cat(tensors=(a, b), dim=0))
print(torch.cat(tensors=(a, b), dim=1))

# GPU로 계산
a = torch.Tensor(np.array([[1, 1, 1, 1], [2, 2, 2, 2]]))
b = torch.Tensor(np.array([[3, 3, 3, 3], [4, 4, 4, 4]]))
a = a.cuda()
b = b.cuda()
c = a+b