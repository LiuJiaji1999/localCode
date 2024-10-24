import cv2
import os

def extract_frames(video_path, output_folder, frame_rate=1):
    # 打开视频文件
    video_cap = cv2.VideoCapture(video_path)
    
    # 检查视频是否打开成功
    if not video_cap.isOpened():
        print("无法打开视频文件")
        return

    # 获取视频的帧率和总帧数
    fps = video_cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"视频帧率: {fps}")
    print(f"总帧数: {total_frames}")

    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 计算抽帧间隔
    frame_interval = int(fps / frame_rate)
    frame_count = 0
    extracted_count = 0

    while video_cap.isOpened():
        ret, frame = video_cap.read()
        
        if not ret:
            break
        
        # 每隔指定的帧数提取一张图片
        if frame_count % frame_interval == 0:
            # 保存帧为图片
            img_filename = os.path.join(output_folder, f"frame_{extracted_count:04d}_DJI_20240927095133_0001_Z.jpg")
            cv2.imwrite(img_filename, frame)
            print(f"保存图片: {img_filename}")
            extracted_count += 1
        
        frame_count += 1

    video_cap.release()
    print(f"提取完毕，共保存 {extracted_count} 张图片。")

# 示例用法
video_path = "./DJI_20240927095133_0001_Z.MP4"  # 视频文件路径
output_folder = "DJI_20240927095133_0001_Z"  # 保存图片的文件夹
frame_rate = 24  # 每秒钟抽取的帧数
# 
extract_frames(video_path, output_folder, frame_rate)