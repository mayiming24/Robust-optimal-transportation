# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 22:12:26 2022

@author: 20447
"""
from scipy.spatial.distance import cdist
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch import autograd
import ot

real_data_num = 1000 #真实数据数量
batch = 64 #每次运算处理多少数据

def make_data(w, b, data_num): 
    """生成 Y = Xw + b + 噪声的实验数据。数据大小data_num*2"""
    n=int(0.2  *  data_num)
    X  =  torch.rand(data_num)
    Y  =  X*w  +  b
    Y[:n,]  += 3
    X  =  X.view(len(X),1)
    Y  =  Y.view(len(Y),1)
    data  =  torch.cat((X,  Y),  dim=1)
    return  data
def make_real_data(w,b,data_num):
    X  =  torch.rand(data_num)
    Y  =  X*w  +  b  #+ 0.1*torch.rand(data_num)
    X  =  X.view(len(X),1)
    Y  =  Y.view(len(Y),1)
    data  =  torch.cat((X,  Y),  dim=1)
    return  data

real_data = make_data(w = 1, b = 1, data_num = 1000) #生成真实数据样本

G = nn.Sequential( #定义生成器，
    nn.Linear(2,64),
    nn.ReLU(),
    nn.Linear(64,2)
)
D = nn.Sequential( #定义判别器
    nn.Linear(2,64),
    nn.ReLU(),
    nn.Linear(64,1),
    #nn.Sigmoid()
)
U=nn.Sequential(
    nn.Linear(2,64),
    nn.ReLU(),
    nn.Linear(64,1)
)

optimizer_G = torch.optim.RMSprop(G.parameters(),lr=0.0001) #定义生成器优化函数
optimizer_D = torch.optim.RMSprop(D.parameters(),lr=0.0001) #定义判别器优化函数
optimizer_U = torch.optim.RMSprop(U.parameters(),lr=0.0001)


GD = np.zeros((80001,2))
k=0.01
for step in range(80001): #运算10001次
    
    """更新5次判别器D"""
    for stepp in range(5):
        A = np.arange(1,1000)
        a = np.random.choice(A, 64)
        real_data_sample = real_data[a] #上面三行对真实数据采样

        noise_z =  torch.Tensor(np.random.rand(64,2)) #产生随机向量
        G_make = G(noise_z) #随机向量丢进G生成
        
        alpha = torch.rand((64, 1))
        data_hat = alpha * real_data_sample + (1 - alpha) *G_make
        #data_hat.requires_grad = True
        pred_hat = D(data_hat)
        gradients =autograd.grad(outputs=pred_hat, inputs=data_hat, grad_outputs=torch.ones(pred_hat.size()),
                                create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradient_penalty =  ((gradients.view(gradients.size()[0], -1).norm(2, 1) - 1) ** 2).mean()
    

        pro_atrist0 = torch.mul(D(real_data_sample),U(real_data_sample)) #给真值打分
        pro_atrist1 = D(G_make)#给假值打分

        G_loss = -torch.mean(pro_atrist1)
        D_loss = -torch.mean(pro_atrist0-pro_atrist1) # + 1*gradient_penalty

        optimizer_D.zero_grad()
        D_loss.backward( )
        optimizer_D.step()

        #for p in D.parameters():
           # p.data.clamp_(-k, k)


    """更新1次生成器G"""
    A = np.arange(1,1000)
    a = np.random.choice(A, 64)
    real_data_sample = real_data[a] #上面三行实现对真实数据采样

    noise_z =  torch.Tensor(np.random.rand(64,2)) #产生随机向量
    G_make = G(noise_z) #随机向量丢进G生成

    pro_atrist1 = D(G_make)#给假值打分

    G_loss = -torch.mean(pro_atrist1)

    optimizer_G.zero_grad()
    G_loss.backward(retain_graph=True)

    optimizer_G.step()
    
    pro_atrist0 = torch.mul(D(real_data_sample),U(real_data_sample))
    pro_atrist2 = 0.6* torch.sum(torch.abs(torch.tensor(np.ones(64))-U(real_data_sample)))
    U_loss = torch.mean(pro_atrist0) + pro_atrist2
    
    optimizer_U.zero_grad()
    U_loss.backward()
    optimizer_U.step()
    
    
    

    GD[step-1][0] = pro_atrist0.data.numpy().mean() #储存生成器得分
    GD[step-1][1] = pro_atrist1.data.numpy().mean() #储存判别器得分

    plt.ion()#下面都是输出画图使的
    if step % 600 == 0:
         print(step)
         noise_z = torch.Tensor(np.random.rand(200,2))
         G_make = G(noise_z)
         real= make_real_data(w = 1, b = 1, data_num = 200)
         C1=cdist(real.detach().numpy() ,G_make.detach().numpy(),metric='euclidean')       
         a=b=np.ones(200)/200
         e=ot.emd2(a,b,C1)
         print(e)
         A = G_make.detach().numpy()
         plt.scatter(real[:,0],real[:,1],c='blue',s=10,marker='^')
         plt.scatter(A[:,0], A[:,1], s=10,c='red',marker='s')
         plt.pause(0.1)