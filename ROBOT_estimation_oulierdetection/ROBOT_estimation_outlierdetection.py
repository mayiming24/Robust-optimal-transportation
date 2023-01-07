# -*- coding: utf-8 -*-
import numpy as np 
import ot
from sklearn.linear_model import LinearRegression
#import Adam
import matplotlib.pyplot as plt
import math
#cost function
import pandas as pd
from scipy.spatial.distance import cdist

def get_data(n,per,err):#n is the number of samples, per is the propotion of outliers, err is the size of outliers 
    n1=int(per*n)
    x=10*np.random.uniform(size=(n,1))
    y=1*x+1+ np.random.normal(0,1,size=(n,1))
    for i in range(n1):
        #y[i,0]+= err 
        y[i,0]+= (err + np.random.normal(x[i,0],1,size=(1)) )      
    return x,y

def get_deviatation (w, x, y):
    deviatation = np.sqrt(np.mean(np.square(w[0]*x+w[1]-y)))
    if math.isnan(deviatation):
        deviatation=0.1
    return deviatation

def get_real_deviatation (w, x, y, per):
    n1 =int(per* x.shape[0] )
    x=x[n1:,]
    y=y[n1:,]
    deviatation = np.mean(np.square(w[0]*x+w[1]-y))
    return deviatation


def ROBOT(x , y , maxiter = 20,reg=0.04,learning_rate=0.04,lambda_val = 1):
    n = x.shape[0]
    g=LinearRegression()
    g=g.fit(x,y)
    w=np.array([g.coef_[0,0],g.intercept_[0]]).reshape((2,1)) 
    sigma=get_deviatation(w, x, y)
    iter = 0
    #sigma=get_deviatation(w, x, y)
    
    while iter < maxiter:  
        n = x.shape[0]   
        wa=np.ones((n,))
        wb=np.ones((n,))
        z=np.random.normal(0,sigma,size=(n,1))
        resid=w[0]*x+w[1]-y
        cost_vector = cdist(resid, z,metric='sqeuclidean')
        bad_set = np.squeeze(cost_vector > 2 * lambda_val)
        cost_vector[cost_vector > 2 * lambda_val] = 2 * lambda_val
        ot_matrix=ot.emd( wa, wb , cost_vector)
        s=np.ones(n)
        for i in range(n):
            temp=ot_matrix[i,]
            stemp = 1-sum(temp[bad_set[i,]])
            s[i] = (abs(stemp)< 1/n**2)
        s = [bool(s[i]) for i in range(n)]
        t = [not s[i] for i in range(n)]
        
        #g=LinearRegression()
        g=g.fit(x[t],y[t])
        w=np.array([g.coef_[0,0],g.intercept_[0]]).reshape((2,1)) 
        sigma=get_deviatation(w, x[t], y[t])       
        iter += 1
        TT=[]
        TF=[]
        FT=[]
        FF=[]
            
        for i in range(n):
            if s[i]:
                if i<= int(100):
                    TT.append(1)#TT
                    TF.append(0)
                    FT.append(0)
                    FF.append(0)
                else:
                    TT.append(0)#TF
                    TF.append(1)
                    FT.append(0)
                    FF.append(0)
            else:
                if i<= int(100):
                    TT.append(0)#FT
                    TF.append(0)
                    FT.append(1)
                    FF.append(0)
                else:
                    TT.append(0)#FF
                    TF.append(0)
                    FT.append(0)
                    FF.append(1)
        TT = [bool(TT[i]) for i in range(n)]
        TF = [bool(TF[i]) for i in range(n)]
        FT = [bool(FT[i]) for i in range(n)]
        FF = [bool(FF[i]) for i in range(n)]
        
        if iter%1 == 0:
            plt.ion()
            plt.scatter(x[TT],y[TT],s=15,c='red',marker='o')
            plt.scatter(x[TF],y[TF],s=15,c='black',marker='v')
            plt.scatter(x[FT],y[FT],s=15,c='blue',marker='p')
            plt.scatter(x[FF],y[FF],s=15,c='green',marker='s')
            plt.pause(0.1)                        
    return w

s=7
x,y=get_data(500,0.2,6)


w=ROBOT(x, y)   


