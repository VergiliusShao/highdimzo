from scipy.special import lambertw
from scipy.optimize import OptimizeResult
import numpy as np
import numpy.linalg as lina
class ExpGrad(object):
    def __init__(self, func, func_p, x0, lower, upper, l1=1.0, l2=1.0,eta=1.0):
        self.func = func
        self.func_p=func_p
        self.x= np.zeros(shape=x0.shape)
        self.x[:]= x0
        self.d = self.x.size
        self.lower=lower
        self.upper=upper
        self.l1=l1
        self.l2=l2
        self.eta=eta
        
    def update(self):
        g=self.func_p(self.x)
        self.step(g)
        return self.x
        
    def step(self,g):
        self.md(g)
        
    def md(self,g):
        beta = 1.0 / self.d
        eta_t=1.0/self.eta
        z=(np.log(np.abs(self.x) / beta + 1.0)) * np.sign(self.x) - g/eta_t
        x_sgn = np.sign(z)
        if self.l2 == 0.0:
            x_val = beta * np.exp(np.maximum(np.abs(z) - self.l1 /eta_t,0.0)) - beta
        else:
            a = beta
            b = self.l2 /eta_t
            c = np.minimum(self.l1 /eta_t - np.abs(z),0.0)
            abc=-c+np.log(a*b)+a*b
            x_val = np.where(abc>=15.0,np.log(abc)-np.log(np.log(abc))+np.log(np.log(abc))/np.log(abc), lambertw( np.exp(abc), k=0).real )/b-a
        x = x_sgn * x_val
        self.x = np.clip(x, self.lower, self.upper)
        

def fmin(func,x0, upper,lower,l1=1.0,l2=1.0,batch=50, maxfev=50,callback=None, epoch_size=10,eta=1.0):
    delta=np.sqrt(2*np.e *(2*np.log(x0.size-1)))/np.sqrt(maxfev)/x0.size
    func_p=Grad_2p_batch(func,batch,delta)
    alg=ExpGrad(func=func,func_p=func_p,x0=x0,upper=upper,lower=lower,l1=l1,l2=l2,eta=eta)
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
        d=x.size
        batch=np.zeros(shape=x.shape)
        batch_v=np.zeros(shape=x.shape)
        batch[:]=x
        batch_v[:]=x
        g=np.zeros(shape=x.shape)
        for i in range(self.n):
            v= np.random.choice([-1.0,1.0],size=x.shape,p=[0.5,0.5])
            batch=np.append(batch,x+self.delta*v, axis=0)
            batch_v=np.append(batch_v,v, axis=0)
        batch_y=self.func(batch)
        tilde_f_x_r= batch_y[0]
        for i in range(1,self.n+1):
            tilde_f_x_l= batch_y[i]
            g+=1.0/self.delta/self.n*(tilde_f_x_l-tilde_f_x_r)*batch_v[i]
        return g
