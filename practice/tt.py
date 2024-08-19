import codecs
import os
import torch
# with open('example.txt', 'a') as file:
#     file.write('Hello, Python!')

# with open('example.txt', 'a+') as file:
#     file.write('\nAppend this line.')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.hub.load('ultralytics/ultralytics', 'yolov8m' ,pretrained=True).to(device)