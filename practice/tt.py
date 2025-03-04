import codecs
import os
import torch
# with open('example.txt', 'a') as file:
#     file.write('Hello, Python!')

# with open('example.txt', 'a+') as file:
#     file.write('\nAppend this line.')

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = torch.hub.load('ultralytics/ultralytics', 'yolov8m' ,pretrained=True).to(device)

# feas_s = torch.tensor([[[-0.2769],
#                         [-0.2704],
#                         [-0.2781],
#                         [-0.2672],
#                         [-0.2640],
#                         [-0.2238]]])
# print(feas_s.shape)
# mean_feas_s = torch.mean(feas_s, 1, keepdim=True)
# print(mean_feas_s)
# xm = mean_feas_s - feas_s
# xc = xm.transpose(1, 2) @ xm 
# print(xc)

# import torch
# tensor = torch.tensor([[1, 2, 3, 4, 5, 6],
#                        [7, 8, 9, 10, 11, 12]])
# print(tensor[..., 4:])
# print(tensor[..., -4:])
# print(tensor[..., :-4])


import torch
import numpy as np
# 假设 targets 是一个包含多个 PyTorch 张量的列表
targets = [
    torch.tensor([[1, 2], [3, 4]]),
    torch.tensor([[5, 6], [7, 8]])
]

# 使用 torch.cat 拼接张量，然后转换为 NumPy 数组
result = torch.cat(targets, 0).numpy()
print(result)
print(result.shape) # (4, 2)
# [[1 2]
#  [3 4]
#  [5 6]
#  [7 8]]


# 使用 np.array 直接转换为 NumPy 数组
result = np.array(targets)
print(result)
print(result.shape) # (2,2,2)
# [[[1 2]
#   [3 4]]

#  [[5 6]
#   [7 8]]]
