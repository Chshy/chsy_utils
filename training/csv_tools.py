import os
import csv

def ensure_directory_exists(file_path):
    """确保目录存在，如果不存在则创建它"""
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def write_to_csv(csv_fname, epoch, phase, epoch_loss, epoch_acc):
    """写入数据到CSV文件，确保文件和目录存在"""
    # 确保目录存在
    ensure_directory_exists(csv_fname)
    
    # 检查文件是否存在，如果不存在则创建并写入标题行
    file_exists = os.path.isfile(csv_fname)
    
    with open(csv_fname, 'a', newline='') as csvfile:
        fieldnames = ['epoch', 'phase', 'loss', 'accuracy']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # 如果文件不存在，写入标题行
        if not file_exists:
            writer.writeheader()
        
        # 写入数据行
        writer.writerow({
            'epoch': epoch,
            'phase': phase,
            'loss': epoch_loss,
            'accuracy': epoch_acc.item()
        })
