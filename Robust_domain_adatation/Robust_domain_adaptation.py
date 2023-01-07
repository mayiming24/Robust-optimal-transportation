# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 21:29:48 2023

@author: yiming ma
"""

import numpy as np
from scipy.spatial.distance import cdist 
import sklearn
import ot
import scipy as sp
from sklearn.model_selection import KFold
import pylab as pl







class Classifier:    
    # cross validate parameters with k-fold classification
   
    
    def crossval(self,X,Y,kerneltype='linear',nbsplits=5,g_range=np.logspace(-3,3,7),l_range = np.logspace(-3,0,4)):
        kf = KFold(n_splits=nbsplits)
        if kerneltype=='rbf':
            dim=(len(g_range),len(l_range))
            results = np.zeros(dim)
            kf = KFold(n_splits=nbsplits, shuffle=True)
    
            for i,g in enumerate(g_range):
                for j,l in enumerate(l_range):
                    self.lambd=l
                    for train, test in kf.split(X):
                        K=sklearn.metrics.pairwise.rbf_kernel(X[train,:],gamma=g)
                        Kt=sklearn.metrics.pairwise.rbf_kernel(X[train,:],X[test,:],gamma=g)
                        self.fit(K,Y[train,:])
                        ypred=self.predict(Kt.T)
                        
                        ydec=np.argmax(ypred,1)
                        yt=np.argmax(Y[test,:],1)

                        results[i,j] += np.mean(ydec==yt)
            results = results /nbsplits
            #print results
    
            i,j = np.unravel_index(results.argmax(), dim)
            
            self.lambd=l_range[j]

            return g_range[i],l_range[j]
        else:
            dim=(len(l_range))
            results = np.zeros(dim)
            kf = KFold(n_splits=nbsplits, shuffle=True)
            for i,l in enumerate(l_range):
                    self.lambd=l
                    for train, test in kf.split(X):
                        K=sklearn.metrics.pairwise.linear_kernel(X[train,:])
                        Kt=sklearn.metrics.pairwise.linear_kernel(X[train,:],X[test,:])
                        self.fit(K,Y[train,:])
                        ypred=self.predict(Kt.T)
                        ydec=np.argmax(ypred,1)
                        yt=np.argmax(Y[test,:],1)
                        results[i] += np.mean(ydec==yt)
            results = results /nbsplits

    
            self.lambd=l_range[results.argmax()]

            return self.lambd



  





class KRRClassifier(Classifier):

    def __init__(self,lambd=1e-2):
        self.lambd=lambd

    def fit(self,K,y,sw=False):
        ns=K.shape[0]
        if sw:
            K=K*sw
        K0=np.vstack((np.hstack((np.eye(ns),np.zeros((ns,1)))),np.zeros((1,ns+1))))
        
        ## true reg in RKHS
        #K0=np.vstack((np.hstack((K,np.zeros((ns,1)))),np.zeros((1,ns+1))))

        K1=np.hstack((K,np.ones((ns,1))))
        if sw:
            y1=K1.T.dot(y*sw)
        else:
            y1=K1.T.dot(y)

        temp=np.linalg.solve(K1.T.dot(K1) + self.lambd*K0,y1)
        self.w,self.b=temp[:-1],temp[-1]

    def predict(self,K):
        return np.dot(K,self.w)+self.b




def jdot_krr(X,y,Xtest,gamma_g=1, numIterBCD = 10, alpha=1,lambd=1e1, 
             method='emd',reg=1,ktype='linear'):
    # Initializations
    n = X.shape[0]
    ntest = Xtest.shape[0]
    wa=np.ones((n,))/n
    wb=np.ones((ntest,))/ntest

    # original loss
    C0=cdist(X,Xtest,metric='sqeuclidean')
    #print np.max(C0)
    C0=C0/np.median(C0)

    # classifier    
    g = KRRClassifier(lambd)

    # compute kernels
    if ktype=='rbf':
        Kt=sklearn.metrics.pairwise.rbf_kernel(Xtest,Xtest,gamma=gamma_g)
    else:
        Kt=sklearn.metrics.pairwise.linear_kernel(Xtest,Xtest)

    C = alpha*C0#+ cdist(y,ypred,metric='sqeuclidean')
    k=0
    while (k<numIterBCD):# and not changeLabels:
        k=k+1
        #print(C)
        if method=='sinkhorn':
            G = ot.sinkhorn(wa,wb,C,reg)
        if method=='emd':
            G=  ot.emd(wa,wb,C)
        #print(G)

        Yst=ntest*G.T.dot(y)
        #print(Yst)

        g.fit(Kt,Yst)
        ypred=g.predict(Kt)
        #print(Yst[:10])
       
        # function cost
        fcost = cdist(y,ypred,metric='sqeuclidean')

        C=alpha*C0+fcost
            
    return g,np.sum(G*(fcost))   

def row_normalize_matrix(matrix):
    """Row-normalize feature matrix"""
    rowsum = np.sum(matrix, axis=1, dtype=np.float32)
    r_inv = np.power(rowsum, -1)
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.sparse.diags(r_inv)
    return r_mat_inv.dot(matrix)


def jdot_krr_robust(X,y,Xtest,gamma_g=1, numIterBCD = 10, alpha=1,lambd=1e1, 
             method='emd',reg=1,ktype='linear',Lambda=1):
    # Initializations
    n = X.shape[0]
    ntest = Xtest.shape[0]
    wa=np.ones((n,))/n
    wb=np.ones((ntest,))/ntest

    # original loss
    C0=cdist(X,Xtest,metric='sqeuclidean')
    #print np.max(C0)
    C0=C0/np.median(C0)

    # classifier    
    g = KRRClassifier(lambd)

    # compute kernels
    if ktype=='rbf':
        Kt=sklearn.metrics.pairwise.rbf_kernel(Xtest,Xtest,gamma=gamma_g)
    else:
        Kt=sklearn.metrics.pairwise.linear_kernel(Xtest,Xtest)

    C = alpha*C0#+ cdist(y,ypred,metric='sqeuclidean')
    k=0
    while (k<numIterBCD):# and not changeLabels:
        Kt_new=Kt
        C_new=C
        C[C > Lambda] = Lambda
        k=k+1
        if method=='sinkhorn':
            G = ot.sinkhorn(wa,wb,C,reg)
        if method=='emd':
            G=  ot.emd(wa,wb,C)
            
        s1 = np.zeros(len(y))
        for i in range(len(y)):
            s1[i] = G[i, np.where(C_new[i, :] > Lambda)[0]].sum()
            u=s1+np.ones(len(y))/len(y)
        #print(C_new)
        v=np.ones([len(y),1])
        v[np.where(abs(u)<0.0007)[0]]=0
        np.delete(G,np.where(v==0) ,axis=0)
        np.delete(y,np.where(v==0) ,axis=0)
        G=row_normalize_matrix(G.T)
            
      
        Yst=G.dot(y)
        
       #Yst=Yst*v

        #print(Yst.shape,v.shape)
        #np.delete(Kt_new,np.where(v==0) ,axis=0)
        #np.delete(Yst, np.where(v==0))

        g.fit(Kt_new,Yst)
        ypred=g.predict(Kt)
        #print(Yst[:10])
        
       
        # function cost
        fcost = cdist(y,ypred,metric='sqeuclidean')
        C=alpha*C0+fcost
        
    return g,np.sum(G*(fcost)) 



seed=198
np.random.seed(seed)

n = 200
ntest=200


def get_data(n,ntest):

    n2=int(n/2)
    n3=int(n/10)
    sigma=0.05
    
    xs=np.random.randn(n,1)+2
    xs[:n2,:]-=4
    ys=sigma*np.random.randn(n,1)+np.sin(xs/2)
    ys[n-n3:,:]+=3
    
    
    
    
    xt=np.random.randn(n,1)+2
    #xt[:n2,:]/=2 
    xt[:n2,:]-=3
      
    yt=sigma*np.random.randn(n,1)+np.sin(xt/2)
    xt+=2
    
    return xs,ys,xt,yt

xs,ys,xt,yt=get_data(n,ntest)


fs_s = lambda x: np.sin(x/2)
fs_t = lambda x: np.sin((x-2)/2)

                       
txvisu=np.linspace(-2,6.5,100)
sxvisu=np.linspace(-4,4,100)
xvisu=np.linspace(-4,6.5,100)
'''
pl.figure(1)
pl.clf()

pl.subplot()
pl.scatter(xs,ys,label='Source samples',edgecolors='k')
pl.scatter(xt,yt,label='Target samples',edgecolors='k')
pl.plot(sxvisu,fs_s(sxvisu),'b',label='Source model')
pl.plot(txvisu,fs_t(txvisu),'g',label='Target model')
pl.xlabel('x')

pl.ylabel('y')
pl.legend(fontsize=7,edgecolor="black",loc='upper left', frameon=True)
pl.title('Toy regression example')
#pl.savefig('imgs/visu_data_reg.eps')
'''

lambd0=10
itermax=15
gamma=1e-1
alpha=0.3
C0=cdist(xs,xt,metric='sqeuclidean')
#print np.max(C0)
C0=C0/np.median(C0)
fcost = cdist(ys,yt,metric='sqeuclidean')
C=alpha*C0+fcost
G0=ot.emd(ot.unif(n),ot.unif(n),C)

model,loss = jdot_krr(xs,ys,xt,gamma_g=gamma,numIterBCD = 100, alpha=alpha, 
                                  lambd=lambd0,ktype='rbf')

modelr,lossr = jdot_krr_robust(xs,ys,xt,gamma_g=gamma,numIterBCD = 100, alpha=alpha, 
                                  lambd=lambd0,ktype='rbf')

g = KRRClassifier(10)
Kt=sklearn.metrics.pairwise.rbf_kernel(xs,xs,gamma=0.1)
g.fit(Kt,ys)



K=sklearn.metrics.pairwise.rbf_kernel(xt,xt,gamma=gamma)
Kvisu=sklearn.metrics.pairwise.rbf_kernel(txvisu.reshape((-1,1)),xt,gamma=gamma)
Kvisu1=sklearn.metrics.pairwise.rbf_kernel(txvisu.reshape((-1,1)),xs,gamma=gamma)
ypred=model.predict(Kvisu)
yorin=g.predict(Kvisu1)
ypredr=modelr.predict(Kvisu)
ypred0=model.predict(K)


# compute true OT

C0=cdist(xs,xt,metric='sqeuclidean')
#print np.max(C0)
C0=C0/np.median(C0)
fcost = cdist(ys,ypred0,metric='sqeuclidean')
C=alpha*C0+fcost
G=ot.emd2(ot.unif(n),ot.unif(n),C)



pl.figure(2)
pl.clf()
pl.scatter(xs,ys,label='Source samples',edgecolors='k',s=10)
pl.scatter(xt,yt,label='Target samples',edgecolors='k',s=10)
'''
pl.plot(sxvisu,fs_s(sxvisu),'b',label='Source model')
pl.plot(txvisu,fs_t(txvisu),'g',label='Target model')
'''
pl.plot(txvisu,yorin,'gray',label='original model')
pl.plot(txvisu,ypred,'blue',label='OT model')

pl.plot(txvisu,ypredr ,'r',label='Robust-OT model')

pl.xlabel('x')

pl.ylabel('y')
pl.legend(fontsize=7,edgecolor="black",loc='upper left', frameon=True)


