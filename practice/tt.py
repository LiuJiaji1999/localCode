import codecs
import os
import torch
# with open('example.txt', 'a') as file:
#     file.write('Hello, Python!')

# with open('example.txt', 'a+') as file:
#     file.write('\nAppend this line.')

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = torch.hub.load('ultralytics/ultralytics', 'yolov8m' ,pretrained=True).to(device)

feas_s = torch.tensor([[[-0.2769],
                        [-0.2704],
                        [-0.2781],
                        [-0.2672],
                        [-0.2640],
                        [-0.2238]]])
print(feas_s.shape)
mean_feas_s = torch.mean(feas_s, 1, keepdim=True)
print(mean_feas_s)
xm = mean_feas_s - feas_s
xc = xm.transpose(1, 2) @ xm 
print(xc)