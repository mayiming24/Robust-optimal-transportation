# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 22:11:01 2022

@author: 20447
"""
from scipy.spatial.distance import cdist
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import ot
real_data_num = 1000 
batch = 64 

def make_data(w, b, data_num,p=0.1,err=3): #p is the proportion of outliers, err is the size of outliers
    n=int(p  *  data_num)
    X  =  torch.rand(data_num)
    Y  =  X*w  +  b
    Y[:n,]  += err
    X  =  X.view(len(X),1)
    Y  =  Y.view(len(Y),1)
    data  =  torch.cat((X,  Y),  dim=1)
    return  data


def make_real_data(w,b,data_num):
    X  =  torch.rand(data_num)
    Y  =  X*w  +  b  
    X  =  X.view(len(X),1)
    Y  =  Y.view(len(Y),1)
    data  =  torch.cat((X,  Y),  dim=1)
    return  data


real_data = make_data(w = 1, b = 1, data_num = 1000) 

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

optimizer_G = torch.optim.RMSprop(G.parameters(),lr=0.0001) 
optimizer_D = torch.optim.RMSprop(D.parameters(),lr=0.0001) 

k=0.01 #the clipping parameter


for step in range(40001): #运算10001次
    
    """更新5次判别器D"""
    for stepp in range(5):
        A = np.arange(1,1000)
        a = np.random.choice(A, 64)
        real_data_sample = real_data[a] 

        noise_z =  torch.Tensor(np.random.rand(64,2)) 
        G_make = G(noise_z) 
        
        '''WGAN-gp
        alpha = torch.rand((64, 1))
        data_hat = alpha * real_data_sample + (1 - alpha) *G_make
        #data_hat.requires_grad = True
        pred_hat = D(data_hat)
        gradients =autograd.grad(outputs=pred_hat, inputs=data_hat, grad_outputs=torch.ones(pred_hat.size()),
                                create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradient_penalty =  ((gradients.view(gradients.size()[0], -1).norm(2, 1) - 1) ** 2).mean()
        '''

        pro_atrist0 = D(real_data_sample)
        pro_atrist1 = D(G_make)

        G_loss = -torch.mean(pro_atrist1)
        D_loss = -torch.mean(pro_atrist0-pro_atrist1) # + 1*gradient_penalty

        optimizer_D.zero_grad()
        D_loss.backward( retain_graph=True )
        optimizer_D.step()

        for p in D.parameters():
            p.data.clamp_(-k, k)


    A = np.arange(1,1000)
    a = np.random.choice(A, 64)
    real_data_sample = real_data[a] 

    noise_z =  torch.Tensor(np.random.rand(64,2)) 
    G_make = G(noise_z) 


    pro_atrist1 = D(G_make)

    G_loss = -torch.mean(pro_atrist1)

    optimizer_G.zero_grad()
    G_loss.backward(retain_graph=True)

    optimizer_G.step()

    plt.ion()
    if step % 600 == 0:
        print(step)
        noise_z = torch.Tensor(np.random.rand(200,2))
        G_make = G(noise_z)
        real= make_real_data(w = 1, b = 1, data_num = 200)
        C1=cdist(real.detach().numpy() ,G_make.detach().numpy(),metric='euclidean')       
        a=b=np.ones(200)/200
        e=ot.emd2(a,b,C1)
        print(e) #W_1 distance
        A = G_make.detach().numpy()
        plt.scatter(real[:,0],real[:,1],c='blue',s=10,marker='^')
        plt.scatter(A[:,0], A[:,1], s=10,c='red',marker='s')
        plt.pause(0.1)
        
