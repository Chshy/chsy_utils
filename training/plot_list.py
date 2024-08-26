import matplotlib.pyplot as plt
import os

def plot_and_save(data_list, labels, ylabel, filename):
    """
    记录多个数据集并保存为图片文件
    
    参数:
    - data_list: 包含需要记录的多个数据集的列表，每个数据集是一个列表或数组
    - labels: 对应每个数据集的标签列表
    - ylabel: y 轴的标签
    - filename: 保存图片的文件名
    """
    plt.figure(figsize=(10, 5))
    
    # 绘制每个数据集
    for data, label in zip(data_list, labels):
        plt.plot(data, label=label)
    
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.title(f'{ylabel} Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)  # 保存为图片文件
    plt.close()

# data1 = [1, 2, 3, 4, 5]
# data2 = [2, 3, 4, 5, 6]
# plot_and_save([data1, data2], ['Data 1', 'Data 2'], 'Value', 'plot.png')