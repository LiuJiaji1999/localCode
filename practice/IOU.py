# pythonCopy code

import numpy as np

def traditional_iou(box1, box2):
    # 计算传统IOU
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    iou = intersection / (area1 + area2 - intersection)
    return iou

def w_iou(box1, box2, class_weights):
    # 计算加权IOU
    iou = traditional_iou(box1, box2)
    w_iou = (1 - iou + theta * class_weights) * iou
    return w_iou

def alpha_iou(box1, box2, alpha):
    # 计算Alpha IOU
    iou = traditional_iou(box1, box2)
    alpha_iou = (1 - (1 - iou)**alpha) * iou
    return alpha_iou

def soft_iou(box1, box2, gamma, beta, delta):
    # 计算Soft IOU
    iou = traditional_iou(box1, box2)
    soft_iou = iou**gamma * (np.exp(-beta * (1 - iou)) + delta)
    return soft_iou

def enhanced_iou(box1, box2, lambda_, adjustment_factors):
    # 计算Enhanced IOU
    iou = traditional_iou(box1, box2)
    enhanced_iou = iou - lambda_ * np.sum(adjustment_factors)
    return enhanced_iou

# 示例使用
box1 = [10, 10, 50, 50]
box2 = [20, 20, 60, 60]
class_weights = 2.0
theta = 0.5
alpha = 2
gamma = 0.5
beta = 0.5
delta = 0.5
lambda_ = 0.1
adjustment_factors = [0.1, 0.2, 0.3]

iou_result = traditional_iou(box1, box2)
print("传统IOU:", iou_result)

w_iou_result = w_iou(box1, box2, class_weights)
print("加权IOU:", w_iou_result)

alpha_iou_result = alpha_iou(box1, box2, alpha)
print("Alpha IOU:", alpha_iou_result)

soft_iou_result = soft_iou(box1, box2, gamma, beta, delta)
print("Soft IOU:", soft_iou_result)

enhanced_iou_result = enhanced_iou(box1, box2, lambda_, adjustment_factors)
print("Enhanced IOU:", enhanced_iou_result)