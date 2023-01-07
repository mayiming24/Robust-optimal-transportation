library(parallel)
cl.cores <- detectCores()

f=function(n){
  library("transport")
  library(stabledist)
  set.seed(Sys.time())
  
  rwasserstein<-function(a,b,lambda=10){
    la=length(a)
    lb=length(b)
    wa = rep(1,la)/la
    wb = rep(1,lb)/lb
    A<-t(sapply(a,rep,lb))
    B<-(sapply(b,rep,la))
    d=abs(A-B)
    d[d>lambda]=lambda
    t=transport(wa,wb,costm = d)
    w=sum(d[cbind(t$from,t$to)]*t$mass) 
    return(w)
  }  
  
  DGP = function(mu, n){
    x0 = rstable(n,alpha = 1.1,beta=0,delta = mu)
    return(x0)
  }
  
  
  
  ###modify this function
  Wdist = function(mu, data){
    W=rep(0,k)
    for (i in 1:k) {
      y_temp = DGP(mu, 1000) 
      W[i] = wasserstein1d(y_temp, data)
    }
    return(mean(W))
  }
  
  RWdist = function(mu, data){
    W=rep(0,k)
    for (i in 1:k) {
      y_temp = DGP(mu, 1000) 
      W[i] = rwasserstein(y_temp, data)
    }
    return(mean(W))
  }
  
  ###modify this function uisng MEWE or MERWE
  muHat = function(data){
    n = length(data)
    opt = optimize(Wdist, interval =  c(-10,10), data)
    muhat = opt$minimum
    return(muhat)
  }
  
  rmuHat = function(data){
    n = length(data)
    opt = optimize(RWdist, interval =  c(-10,10), data)
    muhat = opt$minimum
    return(muhat)
  }
  
  ###Simulation settings
  n = 1000
  L = 10
  mu = 0
  sigma = 1  
  p=0.1
  size=4
  k=20
  
  
  
  n1=floor(n*(1-p))
  Y1 = DGP(mu, n1) 
  Y2 = DGP(size, n-n1)
  
  Y=c(Y1,Y2)
  muYy = muHat(Y)
  rmuYy = rmuHat(Y)
  
  
  return(c(rmuYy,muYy))
}
out=1:1000
cl <- makeCluster(12)
results <- parLapply(cl,out,f)
res <- do.call('rbind',results)
stopCluster(cl)


mean(abs(res[,1]))#BIAS of MERWE

mean(abs(res[,2]))#BIAS of MEWE
#MSE
mean(res[,1]^2) #MSE of MERWE

mean(res[,2]^2)#MSE of MEWE

