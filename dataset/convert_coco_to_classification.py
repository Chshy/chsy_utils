# 将COCO格式的Object Detection数据集转换为Classification数据集
# Chsy 2024.8.24
# 2024.10.16 修改

import os
import cv2
import shutil
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

def segments2boxes(segments):
    """Convert segment labels (OBB) to box labels (HBB), i.e. (cls, xy1, xy2, ...) to (cls, xywh)."""
    boxes = []
    for s in segments:
        x, y = s.T  # segment xy
        boxes.append([x.min(), y.min(), x.max(), y.max()])  # xyxy
    return xyxy2xywh(np.array(boxes))  # xywh

def xyxy2xywh(boxes):
    """Convert bounding boxes from xyxy format to xywh format."""
    xywh_boxes = []
    for box in boxes:
        x_min, y_min, x_max, y_max = box

        # fix labels
        x_min = max(x_min, 0)
        y_min = max(y_min, 0)
        x_max = min(x_max, 1)
        y_max = min(y_max, 1)

        w = x_max - x_min
        h = y_max - y_min
        x_center = x_min + w / 2
        y_center = y_min + h / 2
        xywh_boxes.append([x_center, y_center, w, h])
    return np.array(xywh_boxes)

# 检查输出目录是否存在，并提示用户是否覆盖
def confirm_overwrite(directory):
    if os.path.exists(directory):
        response = input(f"Directory '{directory}' already exists. Do you want to overwrite it? (y/n): ").strip().lower()
        if response != 'y':
            print(f"Operation aborted. '{directory}' not overwritten.")
            return False
        else:
            shutil.rmtree(directory, ignore_errors=True)
            print(f"'{directory}' has been removed.")
    return True

def convert_yolo_to_classification(images_dir, labels_dir, output_dir):
    label_files = list(Path(labels_dir).glob("*.txt"))
    for label_file in tqdm(label_files, desc="Processing labels"):
        with open(label_file, "r") as f:
            label_data = f.readlines()

        image_file = images_dir / (label_file.stem + ".png")
        if not image_file.exists():
            image_file = images_dir / (label_file.stem + ".jpg")
        if not image_file.exists():
            continue

        image = cv2.imread(str(image_file))
        h, w, _ = image.shape

        for line in label_data:
            parts = list(map(float, line.split()))
            class_id = int(parts[0])

            if len(parts) == 5:  # HBB format (class_id, x_center, y_center, width, height)
                x_center, y_center, width, height = parts[1:]
                x_center, y_center, width, height = x_center * w, y_center * h, width * w, height * h

                x_min = int(x_center - width / 2)
                y_min = int(y_center - height / 2)
                x_max = int(x_center + width / 2)
                y_max = int(y_center + height / 2)
            elif len(parts) == 9:  # OBB format (class_id, x1, y1, x2, y2, x3, y3, x4, y4)
                points = np.array(parts[1:]).reshape((4, 2)) * [w, h]
                x_min, y_min = points.min(axis=0).astype(int)
                x_max, y_max = points.max(axis=0).astype(int)
            else:
                continue

            # 裁剪图像
            cropped_image = image[y_min:y_max, x_min:x_max]
            if cropped_image.size == 0:
                continue

            # 保存裁剪后的图像
            class_dir = Path(output_dir) / str(class_id)
            class_dir.mkdir(parents=True, exist_ok=True)

            output_image_path = class_dir / f"{label_file.stem}_{x_min}_{y_min}.jpg"
            cv2.imwrite(str(output_image_path), cropped_image)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Convert YOLO dataset to classification dataset")
    
    # 输入公共前缀
    parser.add_argument("input_prefix", type=str, help="Input dataset common prefix (e.g., /path/to/coco)")

    # 输入和输出路径

    is_COCO = False # 对于COCO 手动在此处修改 (也可以调用的时候改参数)
    if(is_COCO):
        parser.add_argument("--train_images_dir", type=str, default="images/train2017", help="Path to training images directory (relative to input_prefix)")
        parser.add_argument("--val_images_dir", type=str, default="images/val2017", help="Path to validation images directory (relative to input_prefix)")
        parser.add_argument("--train_labels_dir", type=str, default="labels/train2017", help="Path to training labels directory (relative to input_prefix)")
        parser.add_argument("--val_labels_dir", type=str, default="labels/val2017", help="Path to validation labels directory (relative to input_prefix)")
    else:
        parser.add_argument("--train_images_dir", type=str, default="images/train", help="Path to training images directory (relative to input_prefix)")
        parser.add_argument("--val_images_dir", type=str, default="images/val", help="Path to validation images directory (relative to input_prefix)")
        parser.add_argument("--train_labels_dir", type=str, default="labels/train", help="Path to training labels directory (relative to input_prefix)")
        parser.add_argument("--val_labels_dir", type=str, default="labels/val", help="Path to validation labels directory (relative to input_prefix)")
    
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (optional, defaults to auto-generated)")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # 拼接路径
    input_prefix = args.input_prefix
    train_images_dir = os.path.join(input_prefix, args.train_images_dir)
    val_images_dir = os.path.join(input_prefix, args.val_images_dir)
    train_labels_dir = os.path.join(input_prefix, args.train_labels_dir)
    val_labels_dir = os.path.join(input_prefix, args.val_labels_dir)
    
    # 设置输出路径
    output_dir = args.output_dir or input_prefix + "-classification"
    
    train_output_dir = os.path.join(output_dir, "train")
    val_output_dir = os.path.join(output_dir, "val")

    # 确认是否覆盖并删除旧目录
    if confirm_overwrite(output_dir):
        os.makedirs(train_output_dir, exist_ok=True)
        os.makedirs(val_output_dir, exist_ok=True)

        # 处理训练集
        convert_yolo_to_classification(Path(train_images_dir), Path(train_labels_dir), train_output_dir)

        # 处理验证集
        convert_yolo_to_classification(Path(val_images_dir), Path(val_labels_dir), val_output_dir)

        print("Convert complete.")


# 数据存储格式: 转换后的数据集将组织成文件夹结构，每个类别一个文件夹，文件夹中包含该类别的所有裁剪图像。
# 这个结构方便使用torchvision.datasets.ImageFolder来加载数据。

# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader

# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor()
# ])

# train_dataset = datasets.ImageFolder(root=train_output_dir, transform=transform)
# val_dataset = datasets.ImageFolder(root=val_output_dir, transform=transform)

# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
