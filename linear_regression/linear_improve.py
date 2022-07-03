import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l

"""
1. 生成数据集
"""
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

"""
2. 读取数据集
"""
def load_array(data_arrays, batch_size, is_train=True):  #@save
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)

"""
3. 定义模型
"""
from torch import nn

# nn.Linear(2, 1) 表示2输入，1输出
# nn.Sequential 是一个容器，将多个层串联在一起。 当给定输入数据时，Sequential实例将数据传入到第一层， 然后将第一层的输出作为第二层的输入，以此类推。 
net = nn.Sequential(nn.Linear(2, 1))

"""
4. 初始化模型参数
"""
# nn.Linear有weight和bias两个参数，还可以使用替换方法normal_和fill_来重写参数值。
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

"""
5. 定义损失函数
"""
# MSELoss使用L2范数计算
loss = nn.MSELoss()

"""
6. 定义优化算法
"""
# 当我们实例化一个SGD实例时，我们要指定优化的参数 （可通过net.parameters()从我们的模型中获得）以及优化算法所需的超参数字典。 
# 小批量随机梯度下降只需要设置lr值，这里设置为0.03。
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

"""
7. 训练
在每个迭代周期里，我们将完整遍历一次数据集（train_data），不停地从中获取一个小批量的输入和相应的标签。 对于每一个小批量，我们会进行以下步骤:
* 通过调用net(X)生成预测并计算损失l（前向传播）。
* 通过进行反向传播来计算梯度。
* 通过调用优化器来更新模型参数。
"""
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X) ,y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')

w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差：', true_b - b)
