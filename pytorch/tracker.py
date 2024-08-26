# 数据分布跟踪器
# Chsy 2024.8.23
# 注意: 只用来跟踪涉及输入数据的计算
# 如果要跟踪量化后的权重, 不用这个方法 因为其实和输入数据无关

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class DistributionTracker:
    def __init__(self, model, layer_name, bins=50, hist_minmax = None, track_mode='forward'):
        self.model = model
        self.layer_name = layer_name
        self.bins = bins
        self.track_mode = track_mode
        
        self.hist_minmax = hist_minmax
        self.reset()  # 初始化

        self.hook = None
        self.grad_hook = None
        
        self.register_hooks()
    
    def reset(self):
        # 清空之前的记录
        self.cumulative_hist_output = torch.zeros(self.bins)
        self.cumulative_hist_grad = torch.zeros(self.bins)
        
        # 重置最小值和最大值
        if self.hist_minmax is not None and len(self.hist_minmax) == 4:
            self.forward_hist_min = self.hist_minmax[0]
            self.forward_hist_max = self.hist_minmax[1]
            self.backward_hist_min = self.hist_minmax[2]
            self.backward_hist_max = self.hist_minmax[3]
        else:
            self.forward_hist_min = None
            self.forward_hist_max = None
            self.backward_hist_min = None
            self.backward_hist_max = None
    
    def hook_fn(self, module, input, output):
        # print("grad received by hook")
        # print(output)
        if self.forward_hist_min is None or self.forward_hist_max is None:
            self.forward_hist_min = output.min().item()
            self.forward_hist_max = output.max().item()
            if self.backward_hist_min == 0:
                self.backward_hist_min -= 0.1
            if self.backward_hist_max == 0:
                self.backward_hist_max += 0.1
        # else:
        #     self.forward_hist_min = min(self.forward_hist_min, output.min().item())
        #     self.forward_hist_max = max(self.forward_hist_max, output.max().item())
        
        hist = torch.histc(output.detach(), bins=self.bins, min=self.forward_hist_min, max=self.forward_hist_max)
        self.cumulative_hist_output += hist.cpu()
    
    # input是靠近上游的!
    def grad_hook_fn(self, module, grad_input, grad_output):
        # print("from grad_hook_fn")
        # print(grad_output[0].shape)
        
        # grad = grad_output[0].detach()
        grad = grad_input[0].detach()
        
        # print("grad received by hook")
        # # print(grad_input[0])
        # print(grad)
        # print(grad.shape)
        # 注意，这里读到的是一个batch求和的结果

        if self.backward_hist_min is None or self.backward_hist_max is None:
            # print("if")
            self.backward_hist_min = grad.min().item()
            self.backward_hist_max = grad.max().item()
            # print(f"backward_hist_min = {self.backward_hist_min}")
            # print(f"backward_hist_max = {self.backward_hist_max}")
            if self.backward_hist_min == 0:
                self.backward_hist_min -= 0.1
            if self.backward_hist_max == 0:
                self.backward_hist_max += 0.1

        # print(f"backward_hist_min = {self.backward_hist_min}")
        # print(f"backward_hist_max = {self.backward_hist_max}")
        # else:
        #     self.backward_hist_min = min(self.backward_hist_min, grad.min().item())
        #     self.backward_hist_max = max(self.backward_hist_max, grad.max().item())
        
        hist = torch.histc(grad, bins=self.bins, min=self.backward_hist_min, max=self.backward_hist_max)
        self.cumulative_hist_grad += hist.cpu()
        # print(hist)
    
    def register_hooks(self):
        layer = dict(self.model.named_modules())[self.layer_name]
        
        if self.track_mode in ['forward', 'both']:
            self.hook = layer.register_forward_hook(self.hook_fn)
        
        if self.track_mode in ['backward', 'both']:
            self.grad_hook = layer.register_full_backward_hook(self.grad_hook_fn)
    
    def remove_hooks(self):
        if self.hook:
            self.hook.remove()
        if self.grad_hook:
            self.grad_hook.remove()
    
    def get_output_distribution(self):
        prob_dist = self.cumulative_hist_output / self.cumulative_hist_output.sum()
        return prob_dist
    
    def get_grad_distribution(self):
        prob_dist = self.cumulative_hist_grad / self.cumulative_hist_grad.sum()
        return prob_dist
    
    def plot_output_distribution(self, img_savepath="./CDF_f.png"):
        if self.track_mode in ['forward', 'both']:
            prob_dist = self.get_output_distribution()
            
            # 使用前向传播的范围绘制直方图
            bin_edges = torch.linspace(self.forward_hist_min, self.forward_hist_max, self.bins + 1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            plt.bar(bin_centers.numpy(), prob_dist.numpy(), width=(bin_edges[1] - bin_edges[0]).item())
            plt.xlabel('Value')
            plt.ylabel('Output Probability')
            plt.grid(True)
            plt.savefig(img_savepath)
            plt.close()
    
    def plot_grad_distribution(self, img_savepath="./CDF_b.png"):
        if self.track_mode in ['backward', 'both']:
            prob_dist = self.get_grad_distribution()
            
            # 使用反向传播的范围绘制直方图
            bin_edges = torch.linspace(self.backward_hist_min, self.backward_hist_max, self.bins + 1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            plt.bar(bin_centers.numpy(), prob_dist.numpy(), width=(bin_edges[1] - bin_edges[0]).item())
            plt.xlabel('Value')
            plt.ylabel('Gradient Probability')
            plt.grid(True)
            plt.savefig(img_savepath)
            plt.close()

 

# 示例用法

# # 定义一个简单的神经网络
# class SimpleNet(nn.Module):
#     def __init__(self):
#         super(SimpleNet, self).__init__()
#         self.fc1 = nn.Linear(10, 50)
#         self.fc2 = nn.Linear(50, 10)
    
#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

# # 实例化网络
# net = SimpleNet()

# # 使用DistributionTracker进行分布跟踪
# # 'forward' 仅跟踪正向传播
# # 'backward' 仅跟踪反向传播
# # 'both' 跟踪正向和反向传播
# tracker = DistributionTracker(net, 'fc1', track_mode='both')

# # 假设你有一个数据集
# data = torch.randn(1000, 10)
# labels = torch.randint(0, 2, (1000,))
# dataset = torch.utils.data.TensorDataset(data, labels)
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# # 定义一个简单的损失函数和优化器
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

# # 训练网络
# for inputs, targets in dataloader:
#     outputs = net(inputs)
#     loss = criterion(outputs, targets)
    
#     optimizer.zero_grad()
#     loss.backward()  # 反向传播，会触发grad_hook_fn
#     optimizer.step()

# # 根据track_mode选择绘制输出和/或梯度的概率分布
# if tracker.track_mode in ['forward', 'both']:
#     tracker.plot_output_distribution()
    
# if tracker.track_mode in ['backward', 'both']:
#     tracker.plot_grad_distribution()

# # 移除钩子
# tracker.remove_hooks()









































# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import matplotlib.pyplot as plt

# class DistributionTracker:
#     def __init__(self, model, layer_name, bins=50, hist_min=-1, hist_max=1):
#         self.model = model
#         self.layer_name = layer_name
#         self.bins = bins
#         self.hist_min = hist_min
#         self.hist_max = hist_max
#         self.cumulative_hist = torch.zeros(bins)
#         self.hook = None
        
#         # 自动注册钩子
#         self.register_hook()
    
#     def hook_fn(self, module, input, output):
#         hist = torch.histc(output.detach(), bins=self.bins, min=self.hist_min, max=self.hist_max)
#         self.cumulative_hist += hist.cpu()
    
#     def register_hook(self):
#         layer = dict(self.model.named_modules())[self.layer_name]
#         self.hook = layer.register_forward_hook(self.hook_fn)
    
#     def remove_hook(self):
#         if self.hook:
#             self.hook.remove()
    
#     def get_probability_distribution(self):
#         prob_dist = self.cumulative_hist / self.cumulative_hist.sum()
#         return prob_dist
    
#     def plot_distribution(self, img_savepath = "./CDF.png"):
#         prob_dist = self.get_probability_distribution()
#         plt.bar(range(self.bins), prob_dist.numpy())
#         plt.xlabel('Bins')
#         plt.ylabel('Probability')
#         # plt.show()
#         plt.grid(True)
#         plt.savefig(img_savepath)
#         plt.close()

# # 示例用法

# # # 定义一个简单的神经网络
# # class SimpleNet(nn.Module):
# #     def __init__(self):
# #         super(SimpleNet, self).__init__()
# #         self.fc1 = nn.Linear(10, 50)
# #         self.fc2 = nn.Linear(50, 10)
    
# #     def forward(self, x):
# #         x = F.relu(self.fc1(x))
# #         x = self.fc2(x)
# #         return x

# # # 实例化网络
# # net = SimpleNet()

# # # 假设你有一个数据集
# # data = torch.randn(1000, 10)
# # labels = torch.randint(0, 2, (1000,))
# # dataset = torch.utils.data.TensorDataset(data, labels)
# # dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# # # 使用DistributionTracker进行分布跟踪
# # tracker = DistributionTracker(net, 'fc1')

# # # 遍历数据集
# # for inputs, _ in dataloader:
# #     outputs = net(inputs)  # 前向传播

# # # 绘制概率分布
# # tracker.plot_distribution()

# # # 移除钩子（在不再需要跟踪时）
# # tracker.remove_hook()
