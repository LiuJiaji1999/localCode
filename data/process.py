import csv
import random
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import cv2
from tqdm import tqdm


# =========================
# 这里改成你的数据路径
# =========================
DATASET_ROOT = Path("/Users/rl/Documents/PhD-student/Untitled_Folder/dataset/data_e")   # 改成你的根目录
ANNOTATIONS_DIR = DATASET_ROOT / "xml"
IMAGES_DIR = DATASET_ROOT / "image"

OUTPUT_DIR = DATASET_ROOT / "output_bad_only"
SAVE_IMAGES_DIR = OUTPUT_DIR / "images"         # 仅保存含 bad 的原图
SAVE_LABELS_DIR = OUTPUT_DIR / "labels"         # 仅保存含 bad 的 YOLO 标签
VIS_SAMPLES_DIR = OUTPUT_DIR / "vis_samples"    # 仅抽样可视化 20 张

# 全部类别（用于统计）
ALL_CLASSES = [
    "nail_good",
    "nail_bad_invalid",
    "nail_bad_drop",
]

# 仅用于 YOLO 转换和可视化的类别
TARGET_CLASSES = [
    "nail_bad_drop",
    "nail_bad_invalid"
]

# YOLO 类别映射
YOLO_CLASS_TO_ID = {
    "nail_bad_drop": 0,
    "nail_bad_invalid": 1
}

# 可视化样本数
NUM_VIS_SAMPLES = 20

# 随机种子，保证每次抽样一致；不需要可复现的话可删掉
RANDOM_SEED = 42


def parse_voc_xml(xml_path):
    """
    解析 VOC XML
    返回:
        img_w, img_h, objects
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    img_w = int(size.findtext("width"))
    img_h = int(size.findtext("height"))

    objects = []
    for obj in root.findall("object"):
        name = obj.findtext("name")
        bndbox = obj.find("bndbox")

        xmin = int(float(bndbox.findtext("xmin")))
        ymin = int(float(bndbox.findtext("ymin")))
        xmax = int(float(bndbox.findtext("xmax")))
        ymax = int(float(bndbox.findtext("ymax")))

        objects.append({
            "name": name,
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
        })

    return img_w, img_h, objects


def voc_box_to_yolo(img_w, img_h, xmin, ymin, xmax, ymax):
    """
    VOC框 -> YOLO格式
    """
    x_center = ((xmin + xmax) / 2.0) / img_w
    y_center = ((ymin + ymax) / 2.0) / img_h
    width = (xmax - xmin) / img_w
    height = (ymax - ymin) / img_h
    
    return x_center, y_center, width, height

    # dw = 1.0 / img_w
    # dh = 1.0 / img_h

    # x = (xmin + xmax) / 2.0
    # y = (ymin + ymax) / 2.0 
    # w = xmax - xmin
    # h = ymax - ymin

    # x = x * dw
    # y = y * dh
    # w = w * dw
    # h = h * dh

    # return x, y, w, h


def get_color_by_class(class_name):
    color_map = {
        "nail_bad_invalid": (0, 0, 255),   # 红色
        "nail_bad_drop": (0, 255, 0),    # 黄色
    }
    return color_map.get(class_name, (0, 255, 0))


def draw_boxes(image, objects):
    """
    仅画 bad 类框
    """
    for obj in objects:
        cls_name = obj["name"]
        if cls_name not in TARGET_CLASSES:
            continue

        xmin, ymin, xmax, ymax = obj["xmin"], obj["ymin"], obj["xmax"], obj["ymax"]
        color = get_color_by_class(cls_name)

        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)

        label = cls_name
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2

        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        text_x = xmin
        text_y = max(ymin - 8, th + 5)

        cv2.rectangle(
            image,
            (text_x, text_y - th - baseline - 4),
            (text_x + tw + 4, text_y + baseline),
            color,
            -1
        )
        cv2.putText(
            image,
            label,
            (text_x + 2, text_y - 2),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA
        )

    return image


def find_image_by_xml_name(images_dir, xml_path):
    """
    根据 XML 文件名本身找对应图片
    例如:
        IMG_8673@0.xml -> IMG_8673@0.jpg / .png / .jpeg
    """
    stem = xml_path.stem
    exts = [".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"]

    for ext in exts:
        img_path = images_dir / f"{stem}{ext}"
        if img_path.exists():
            return img_path

    for p in images_dir.rglob("*"):
        if p.is_file() and p.stem == stem:
            return p

    return None


def main():
    random.seed(RANDOM_SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SAVE_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    SAVE_LABELS_DIR.mkdir(parents=True, exist_ok=True)
    VIS_SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    if not ANNOTATIONS_DIR.exists():
        print(f"标注目录不存在: {ANNOTATIONS_DIR}")
        return

    if not IMAGES_DIR.exists():
        print(f"图片目录不存在: {IMAGES_DIR}")
        return

    xml_files = sorted(ANNOTATIONS_DIR.glob("*.xml"))
    if not xml_files:
        print(f"未找到 XML 文件: {ANNOTATIONS_DIR}")
        return

    # 保存 YOLO 类别文件
    classes_txt = OUTPUT_DIR / "classes.txt"
    with open(classes_txt, "w", encoding="utf-8") as f:
        for cls_name in TARGET_CLASSES:
            f.write(cls_name + "\n")

    # 统计：全部类别
    class_object_count = defaultdict(int)
    class_image_count = defaultdict(int)

    total_xml = 0
    kept_xml = 0
    skipped_xml = 0
    copied_image_count = 0
    total_target_objects = 0
    missing_image_count = 0

    # 用于后续抽样可视化
    vis_candidates = []

    for xml_path in tqdm(xml_files, desc="Processing XMLs", unit="xml"):
        total_xml += 1

        try:
            img_w, img_h, objects = parse_voc_xml(xml_path)
        except Exception as e:
            print(f"\n解析失败: {xml_path.name}, 错误: {e}")
            continue

        # 全类别统计
        appeared_classes_in_image = set()
        for obj in objects:
            cls_name = obj["name"]
            if cls_name in ALL_CLASSES:
                class_object_count[cls_name] += 1
                appeared_classes_in_image.add(cls_name)

        for cls_name in appeared_classes_in_image:
            class_image_count[cls_name] += 1

        # 仅保留 bad 类
        target_objects = [obj for obj in objects if obj["name"] in TARGET_CLASSES]
        if len(target_objects) == 0:
            skipped_xml += 1
            continue

        # 转 YOLO
        yolo_lines = []
        valid_target_objects = []

        for obj in target_objects:
            cls_name = obj["name"]

            xmin = max(0, min(obj["xmin"], img_w))
            ymin = max(0, min(obj["ymin"], img_h))
            xmax = max(0, min(obj["xmax"], img_w))
            ymax = max(0, min(obj["ymax"], img_h))

            if xmax <= xmin or ymax <= ymin:
                print(f"\n警告: 非法框, 跳过 {xml_path.name} -> {obj}")
                continue

            x_center, y_center, width, height = voc_box_to_yolo(
                img_w, img_h, xmin, ymin, xmax, ymax
            )

            class_id = YOLO_CLASS_TO_ID[cls_name]
            yolo_lines.append(
                f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            )
            valid_target_objects.append({
                "name": cls_name,
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
            })

        # 如果 bad 类框都非法，就跳过
        if len(yolo_lines) == 0:
            skipped_xml += 1
            continue

        # 找对应图片
        img_path = find_image_by_xml_name(IMAGES_DIR, xml_path)
        if img_path is None:
            missing_image_count += 1
            print(f"\n警告: 找不到对应图片 -> {xml_path.name}")
            continue

        # 保存原图（仅 bad 样本）
        dst_img_path = SAVE_IMAGES_DIR / img_path.name
        shutil.copy2(img_path, dst_img_path)
        copied_image_count += 1

        # 保存 yolo 标签
        yolo_label_path = SAVE_LABELS_DIR / f"{xml_path.stem}.txt"
        with open(yolo_label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(yolo_lines))

        kept_xml += 1
        total_target_objects += len(valid_target_objects)

        vis_candidates.append({
            "img_path": dst_img_path,
            "xml_name": xml_path.name,
            "objects": valid_target_objects,
        })

    # 抽样可视化
    if len(vis_candidates) > 0:
        sample_num = min(NUM_VIS_SAMPLES, len(vis_candidates))
        sampled_items = random.sample(vis_candidates, sample_num)

        for item in tqdm(sampled_items, desc="Visualizing samples", unit="img"):
            image = cv2.imread(str(item["img_path"]))
            if image is None:
                print(f"\n警告: 图片读取失败 {item['img_path']}")
                continue

            vis_image = draw_boxes(image.copy(), item["objects"])
            save_path = VIS_SAMPLES_DIR / item["img_path"].name
            cv2.imwrite(str(save_path), vis_image)
# # 
    # 保存统计 CSV
    statistics_csv = OUTPUT_DIR / "statistics.csv"
    with open(statistics_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["class_name", "image_count", "object_count", "used_for_yolo_and_vis"])
        for cls_name in ALL_CLASSES:
            writer.writerow([
                cls_name,
                class_image_count[cls_name],
                class_object_count[cls_name],
                "yes" if cls_name in TARGET_CLASSES else "no"
            ])

    print("\n" + "=" * 70)
    print("处理完成（仅保留有效 bad 样本）")
    print(f"XML总数: {total_xml}")
    print(f"保留样本数(含 bad 类): {kept_xml}")
    print(f"跳过样本数(仅 good 或无有效 bad 框): {skipped_xml}")
    print(f"缺失对应图片数: {missing_image_count}")
    print(f"保存 bad 原图数: {copied_image_count}")
    print(f"保存可视化抽样数: {min(NUM_VIS_SAMPLES, len(vis_candidates))}")
    print(f"参与YOLO转换的 bad 目标总数: {total_target_objects}")
    print("-" * 70)
    print("全部类别统计:")
    for cls_name in ALL_CLASSES:
        print(
            f"{cls_name:20s} | 图片数: {class_image_count[cls_name]:6d} | "
            f"目标数: {class_object_count[cls_name]:6d} | "
            f"{'参与转换/可视化' if cls_name in TARGET_CLASSES else '仅统计'}"
        )
    print("-" * 70)
    print(f"bad原图保存目录: {SAVE_IMAGES_DIR}")
    print(f"YOLO标签保存目录: {SAVE_LABELS_DIR}")
    print(f"抽样可视化目录: {VIS_SAMPLES_DIR}")
    print(f"YOLO类别文件: {classes_txt}")
    print(f"统计文件: {statistics_csv}")
    print("=" * 70)


if __name__ == "__main__":
    main()
