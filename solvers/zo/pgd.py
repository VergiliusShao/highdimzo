from scipy.optimize import OptimizeResult
import numpy as np
class PGD(object):
    def __init__(self, func, func_p, x0, lower, upper, l1=1.0, l2=1.0,eta=1.0):
        self.func = func
        self.func_p=func_p
        self.x= np.zeros(shape=x0.shape)
        self.x[:]= x0
        self.d = self.x.size
        self.lower=lower
        self.upper=upper
        self.eta=eta
        self.l1=l1
        self.l2=l2

    def update(self):
        g=self.func_p(self.x)
        self.step(g)
        return self.x
                

    def step(self,g):
        self.md(g)

    def md(self,g):
        alpha = 1.0/self.eta
        z = self.x-g/alpha
        x_sgn = np.sign(z)
        x_val = np.maximum(alpha*np.abs(z)-self.l1,0.0)/(alpha+self.l2)
        x = x_sgn * x_val
        self.x = np.clip(x, self.lower, self.upper)


def fmin(func,x0, upper,lower,finite_sum=False,l1=1.0,l2=1.0, maxfev=50,callback=None, epoch_size=10,eta=1.0):
    delta=1.0/np.sqrt(maxfev*x0.size)
    if finite_sum:
        func_p=Group_Grad_2p(func,maxfev,delta)
        func=Group_Loss(func)
    else:
        func_p=Grad_2p_batch(func,maxfev,delta)
    alg=PGD(func=func,func_p=func_p,x0=x0,upper=upper,lower=lower,l1=l1,l2=l2,eta=eta)
    nit=maxfev
    fev=1
    y=None
    while fev <= maxfev:
        y=alg.update()
        if callback is not None and fev%epoch_size==0:
                res=OptimizeResult(func=func(y), x=y, nit=fev,
                          nfev=fev, success=(y is not None))
                callback(res)
        fev+=1
    return OptimizeResult(func=func(y), x=y, nit=nit,
                          nfev=fev, success=(y is not None))

class Grad_2p_batch:
    def __init__(self, func, n,delta):
        self.func=func
        self.n=n
        self.delta=delta
    def __call__(self, x):
        batch=np.zeros(shape=x.shape)
        batch_v=np.zeros(shape=x.shape)
        batch[:]=x
        batch_v[:]=x
        g=np.zeros(shape=x.shape)
        for i in range(self.n):
            v= np.random.normal(size=x.shape)
            batch=np.append(batch,x+self.delta*v, axis=0)
            batch_v=np.append(batch_v,v, axis=0)
        batch_y=self.func(batch)
        tilde_f_x_r= batch_y[0]
        for i in range(1,self.n):
            tilde_f_x_l= batch_y[i]
            g[0]+=1.0/self.delta/self.n*(tilde_f_x_l-tilde_f_x_r)*batch_v[i]
        return g

class Group_Loss:
    def __init__(self, func_ary):
        self.func_ary=func_ary
    def __call__(self, x):
        loss=0
        for func in self.func_ary:
            loss+=func(x)
        return loss/len(self.func_ary)
    
class Group_Grad_2p:
    def __init__(self, func_ary, n, delta):
        self.func_ary=func_ary
        self.n=n
        self.delta=delta
    def __call__(self, x):
        g=np.zeros(shape=x.shape)
        for i in range(self.n):
            idx =np.random.randint(len(self.func_ary))
            v= np.random.normal(size=x.shape)
            tilde_f_x_l= self.func_ary[idx](x+self.delta*v)
            tilde_f_x_r= self.func_ary[idx](x)
            g+=1.0/self.delta/self.n*(tilde_f_x_l-tilde_f_x_r)*v
        return g
        
