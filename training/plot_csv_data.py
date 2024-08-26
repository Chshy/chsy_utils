import os
import pandas as pd
import matplotlib.pyplot as plt

# from utils.record_and_plot import plot_and_save
from plot_list import plot_and_save

def process_csv_files(folder_path):
    # 用于存储不同图表的数据
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    labels = []

    # 遍历指定文件夹内的所有CSV文件
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            # 获取w, a, g的值
            base_name = os.path.splitext(filename)[0]
            wag_values = base_name.split('(')[-1].split(')')[0]
            labels.append(wag_values)
            
            # 读取CSV文件，使用首行作为列名
            file_path = os.path.join(folder_path, filename)
            print(f"Read file: {file_path}")
            df = pd.read_csv(file_path)
            
            # 分离train和test的数据
            train_df = df[df['phase'] == 'train']
            test_df = df[df['phase'] == 'val']

            # 记录每个CSV文件的训练和测试数据
            train_losses.append(train_df['loss'].values)
            test_losses.append(test_df['loss'].values)
            train_accuracies.append(train_df['accuracy'].values)
            test_accuracies.append(test_df['accuracy'].values)

    # 绘制并保存图表
    plot_and_save(train_losses, labels, 'Train Loss', os.path.join(folder_path, 'train_loss.png'))
    plot_and_save(test_losses, labels, 'Test Loss', os.path.join(folder_path, 'test_loss.png'))
    plot_and_save(train_accuracies, labels, 'Train Accuracy', os.path.join(folder_path, 'train_accuracy.png'))
    plot_and_save(test_accuracies, labels, 'Test Accuracy', os.path.join(folder_path, 'test_accuracy.png'))


if __name__ == "__main__":

    # folder_path = "/home/HD/ycs/resnet-cifar10/0816g1"

    # folder_path = "/home/HD/ycs/resnet-cifar10/runs/train/0823-2355"
    folder_path = "/home/HD/ycs/resnet-cifar10/runs/train/0825-0147"

    process_csv_files(folder_path)
