import cv2
import os

# 输入图片所在的文件夹路径
image_folder = '/Users/rl/Documents/PhD_student/Untitled_Folder/val'

# 定义输出视频的路径和文件名
video_name = 'powerdata.mp4'

# 获取文件夹中的图片文件名列表，并按名称排序
images = [img for img in os.listdir(image_folder) if img.endswith((".png", ".jpg", ".jpeg"))]
images.sort()

# 读取第一张图片来获取视频帧的尺寸
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

# 定义视频编码器及其参数
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'XVID' 也可以用于 .avi 格式
video = cv2.VideoWriter(video_name, fourcc, 30, (width, height))

# 遍历图片文件，逐个写入视频
for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

# 释放视频写入对象
video.release()

print("视频已成功生成:", video_name)
