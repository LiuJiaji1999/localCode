# 合并相同的InsDet-FULL、InsDet-FULL2、InsDet-FULL3、InsDet-FULL4的二三级目录下的所有目录和文件到新的 具有一致二三级目录的新的一级目录InsDet-FULL-Merged下

import os
import shutil
from pathlib import Path

def merge_directories_with_structure(source_dirs, target_dir, sub_dirs=['Background', 'Objects', 'Scenes']):
    """
    合并多个源目录到目标目录，保持一致的二三级目录结构
    
    参数:
        source_dirs: 源目录列表
        target_dir: 目标目录
        sub_dirs: 二级目录名称列表
    """
    # 确保目标目录存在
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建所有二级目录
    for sub_dir in sub_dirs:
        (target_dir / sub_dir).mkdir(exist_ok=True)
    
    # 记录已处理的三级目录，避免重复创建
    processed_third_level_dirs = set()
    
    # 遍历每个源目录
    for source_dir in source_dirs:
        source_dir = Path(source_dir)
        if not source_dir.exists():
            print(f"警告: 源目录 {source_dir} 不存在，跳过")
            continue
            
        print(f"正在处理源目录: {source_dir}")
        
        # 遍历每个二级目录
        for sub_dir in sub_dirs:
            source_sub_dir = source_dir / sub_dir
            if not source_sub_dir.exists():
                print(f"警告: 二级目录 {source_sub_dir} 不存在，跳过")
                continue
                
            target_sub_dir = target_dir / sub_dir
            
            # 遍历三级目录
            for third_level_dir in source_sub_dir.iterdir():
                if not third_level_dir.is_dir():
                    continue
                    
                # 创建目标三级目录（如果尚未创建）
                target_third_level_dir = target_sub_dir / third_level_dir.name
                if target_third_level_dir not in processed_third_level_dirs:
                    target_third_level_dir.mkdir(exist_ok=True)
                    processed_third_level_dirs.add(target_third_level_dir)
                
                # 复制三级目录下的所有内容
                for item in third_level_dir.iterdir():
                    target_item = target_third_level_dir / item.name
                    
                    if item.is_file():
                        # 处理同名文件
                        if target_item.exists():
                            # 添加后缀避免覆盖
                            counter = 1
                            while target_item.exists():
                                stem = item.stem
                                suffix = item.suffix
                                target_item = target_third_level_dir / f"{stem}_{counter}{suffix}"
                                counter += 1
                        
                        shutil.copy2(item, target_item)
                        print(f"复制文件: {item} -> {target_item}")
                    elif item.is_dir():
                        # 处理子目录（四级及更深目录）
                        shutil.copytree(item, target_item, dirs_exist_ok=True)
                        print(f"复制目录: {item} -> {target_item}")

if __name__ == "__main__":
    # 配置源目录和目标目录
    base_dir = '/Users/rl/Documents/PhD-student/Untitled_Folder'
    source_dirs = [
        os.path.join(base_dir, "InsDet-FULL"),
        os.path.join(base_dir, "InsDet-FULL2"),
        os.path.join(base_dir, "InsDet-FULL3"),
        os.path.join(base_dir, "InsDet-FULL4")
    ]
    target_dir = os.path.join('/Users/rl/Desktop', "InsDet-FULL-Merged")
    
    # 如果目标目录已存在，询问用户如何处理
    if os.path.exists(target_dir):
        response = input(f"目标目录 {target_dir} 已存在。要清空它吗？(y/n): ").lower()
        if response == 'y':
            shutil.rmtree(target_dir)
            os.makedirs(target_dir)
            print("目标目录已清空，将创建新的合并目录。")
        else:
            print("将在现有目录基础上继续合并。")
    
    # 执行合并
    merge_directories_with_structure(source_dirs, target_dir)
    
    print("目录合并完成！合并后的目录结构保持一致。")