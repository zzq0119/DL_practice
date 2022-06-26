#!/usr/bin/python3

import random
import torch
from d2l import torch as d2l

def synthetic_data(w, b, num_examples):  #@save
    """生成y=Xw+b+噪声"""
    #指定μ，σ和输出张量的形状，返回tensor
    X = torch.normal(0, 1, (num_examples, len(w)))
    #矩阵乘法
    y = torch.matmul(X, w) + b
    #加上噪声
    y += torch.normal(0, 0.01, y.shape)
    #把y从一个向量变成矩阵，每一行一个元素。-1是通配符
    return X, y.reshape((-1, 1))

def data_iter(batch_size, features, labels):
    """
    训练模型时要对数据集进行遍历，每次抽取一小批量样本，并使用它们来更新我们的模型。 由于这个过程是训练机器学习算法的基础，所以有必要定义一个函数， 该函数能打乱数据集中的样本并以小批量方式获取数据。
    在下面的代码中，我们定义一个data_iter函数， 该函数接收批量大小、特征矩阵和标签向量作为输入，生成大小为batch_size的小批量。 每个小批量包含一组特征和标签。
    """
    num_examples = len(features)
    indices = list(range(num_examples))
    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]

def linreg(X, w, b):  #@save
    """线性回归模型"""
    return torch.matmul(X, w) + b

def squared_loss(y_hat, y):  #@save
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

def sgd(params, lr, batch_size):  #@save
    """小批量随机梯度下降"""
    #torch.no_grad() 是一个上下文管理器，被该语句 wrap 起来的部分将不会track 梯度。这样可以执行计算，但该计算不会在反向传播中被记录。
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

if __name__ == '__main__':
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    #features中的每一行都包含一个二维数据样本， labels中的每一行都包含一维标签值（一个标量）。
    features, labels = synthetic_data(true_w, true_b, 1000)

    print('features:', features[0],'\nlabel:', labels[0])
    #通过生成第二个特征features[:, 1]和labels的散点图， 可以直观观察到两者之间的线性关系。
    d2l.set_figsize()
    d2l.plt.scatter(features[:, (1)].detach().numpy(), labels.detach().numpy(), 1);
    d2l.plt.show()

    w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    batch_size = 10

    for X, y in data_iter(batch_size, features, labels):
        print(X, '\n', y)
        break

    lr = 0.03
    num_epochs = 3
    net = linreg
    loss = squared_loss

    for epoch in range(num_epochs):
        for X, y in data_iter(batch_size, features, labels):
            l = loss(net(X, w, b), y)  # X和y的小批量损失
            # 因为l形状是(batch_size,1)，而不是一个标量。l中的所有元素被加到一起，
            # 并以此计算关于[w,b]的梯度
            l.sum().backward()
            sgd([w, b], lr, batch_size)  # 使用参数的梯度更新参数
        with torch.no_grad():
            train_l = loss(net(features, w, b), labels)
            print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

    print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
    print(f'b的估计误差: {true_b - b}')
