library(parallel)
cl.cores <- detectCores()

f=function(n){
  library("transport")
  
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
  
  DGP = function(mu, n, L, sigma){
    x0 = rnorm(n*L, mu, sigma)
    x = matrix(x0, L, n)
    y0 = exp(x)
    y = apply(y0, MARGIN = 2, FUN = sum)
    return(y)
  }
  
  ###modify this function
  Wdist = function(mu, data, L, sigma){
    W=rep(0,k)
    for (i in 1:k) {
      y_temp = DGP(mu, 1000, L, sigma) 
      W[i] = wasserstein1d(y_temp, data)
    }
    return(mean(W))
  }
  
  RWdist = function(mu, data, L, sigma){
    W=rep(0,k)
    for (i in 1:k) {
      y_temp = DGP(mu, 1000, L, sigma) 
      W[i] = rwasserstein(y_temp, data)
    }
    return(mean(W))
  }
  
  ###modify this function uisng MEWE or MERWE
  muHat = function(data, L, sigma){
    n = length(data)
    opt = optimize(Wdist, interval =  c(-10,10), data, L, sigma)
    muhat = opt$minimum
    return(muhat)
  }
  
  rmuHat = function(data, L, sigma){
    n = length(data)
    opt = optimize(RWdist, interval =  c(-10,10), data, L, sigma)
    muhat = opt$minimum
    return(muhat)
  }
  
  ###Simulation settings
  n = 100
  L = 10
  mu = 0
  sigma = 1  
  p=0.8
  size=1
  k=10
  n1=floor(n*p)
  Y1 = DGP(mu, n1, L, sigma) 
  Y2 = DGP(size, n-n1, L, sigma)
  Y=c(Y1,Y2)
  muYy = muHat(Y, L, sigma)
  rmuYy = rmuHat(Y, L, sigma)
  return(c(rmuYy,muYy))
}

out=1:1000                                      
cl <- makeCluster(12)
results <- parLapply(cl,out,f)
res <- do.call('rbind',results)

stopCluster(cl)

mean(abs(res[,1]))

mean(abs(res[,2]))

mean(res[,1]^2)

mean(res[,2]^2)

