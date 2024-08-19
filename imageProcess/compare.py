import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity
import cv2
import os
import hashlib
import math
import imagehash
 
'''
    粗暴的md5比较 返回是否完全相同
'''
def md5_similarity(img1_path, img2_path):
    file1 = open(img1_path, "rb")
    file2 = open(img2_path, "rb")
    md = hashlib.md5()
    md.update(file1.read())
    res1 = md.hexdigest()
    
    md = hashlib.md5()
    md.update(file2.read())
    res2 = md.hexdigest()
    return res1 == res2
 
def normalize(data):
    return data / np.sum(data)
 
'''
    直方图相似度
    相关性比较 cv2.HISTCMP_CORREL：值越大，相似度越高
    相交性比较 cv2.HISTCMP_INTERSECT：值越大，相似度越高
    卡方比较 cv2.HISTCMP_CHISQR：值越小，相似度越高
    巴氏距离比较 cv2.HISTCMP_BHATTACHARYYA：值越小，相似度越高
'''
def hist_similarity(img1, img2, hist_size=256):
    imghistb1 = cv2.calcHist([img1], [0], None, [hist_size], [0, 256])
    imghistg1 = cv2.calcHist([img1], [1], None, [hist_size], [0, 256])
    imghistr1 = cv2.calcHist([img1], [2], None, [hist_size], [0, 256])
 
    imghistb2 = cv2.calcHist([img2], [0], None, [hist_size], [0, 256])
    imghistg2 = cv2.calcHist([img2], [1], None, [hist_size], [0, 256])
    imghistr2 = cv2.calcHist([img2], [2], None, [hist_size], [0, 256])
 
    distanceb = cv2.compareHist(normalize(imghistb1), normalize(imghistb2), cv2.HISTCMP_CORREL)
    distanceg = cv2.compareHist(normalize(imghistg1), normalize(imghistg2), cv2.HISTCMP_CORREL)
    distancer = cv2.compareHist(normalize(imghistr1), normalize(imghistr2), cv2.HISTCMP_CORREL)
    meandistance = np.mean([distanceb, distanceg, distancer])
    return meandistance
 
def PSNR(img1, img2):
    mse = np.mean((img1/255. - img2/255.) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
 
def SSIM(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # 计算两个灰度图像之间的结构相似度
    score, diff = structural_similarity(gray1, gray2, win_size=101, full=True)
    # diff = (diff * 255).astype("uint8")
    # print("SSIM:{}".format(score))
    return score, diff
 
def ssim(img1, img2):
    u_true = np.mean(img1)
    u_pred = np.mean(img2)
    var_true = np.var(img1)
    var_pred = np.var(img2)
    std_true = np.sqrt(var_true)
    std_pred = np.sqrt(var_pred)
    c1 = np.square(0.01*7)
    c2 = np.square(0.03*7)
    ssim = (2 * u_true * u_pred + c1) * (2 * std_pred * std_true + c2)
    denom = (u_true ** 2 + u_pred ** 2 + c1) * (var_pred + var_true + c2)
    return ssim/denom

def aHash(img1_path,img2_path):
    # 生成图像的感知哈希

    hash1 = imagehash.average_hash(Image.open(img1_path))
    hash2 = imagehash.average_hash(Image.open(img2_path))
    
    # 计算相似度
    aHash_similarity = (hash1 - hash2) / len(hash1.hash) **2 
    return aHash_similarity


def MSE(img1,img2):
    mse = np.mean( (img1 - img2) ** 2 )
    return mse
 
 
if __name__ == '__main__':
    img1_path = './image/000.jpg'
    img2_path = './image/001.jpg'
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
 
    # 1.粗暴的md5比较 返回是否完全相同
    print('md5_similarity:', md5_similarity(img1_path, img2_path))
    # 2.直方图相似度
    print('hist_similarity:', hist_similarity(img1, img2))
    # 3.PSNR
    print('PSNR:', PSNR(img1, img2))
    # 4.SSIM
    print('灰度图SSIM:', SSIM(img1, img2)[0])
    print('RGB图_ssim:', ssim(img1, img2))
    print('RGB图_structural_similarity:', structural_similarity(img1,img2, channel_axis=2))
    # 5.感知哈希
    print('平均值哈希:',aHash(img1_path, img2_path))
    # 6.MSE
    print('MSE:', MSE(img1, img2))
