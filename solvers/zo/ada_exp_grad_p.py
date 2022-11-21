from cmath import inf
from scipy.special import lambertw
from scipy.optimize import OptimizeResult
import numpy as np
import numpy.linalg as lina
class AdaExpGradP(object):
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
        self.eta=0.0
        self.D=1.0

    def update(self):
        g=self.func_p(self.x)
        self.step(g)
        return self.x
        
    def step(self,g):
        self.md(g)
        
    def md(self,g):
        beta = 1.0 / self.d
        eta_t=np.maximum(np.sqrt(self.eta),1)
        z=(np.log(np.abs(self.x) / beta + 1.0)) * np.sign(self.x) - g/eta_t
        v_sgn = np.sign(z)
        if self.l2 == 0.0:
            v_val = beta * np.exp(np.maximum(np.abs(z) - self.l1/eta_t ,0.0)) - beta
        else:
            a = beta
            b = self.l2/eta_t
            c = np.minimum(self.l1/eta_t- np.abs(z),0.0)
            abc=-c+np.log(a*b)+a*b
            v_val = np.where(abc>=15.0,np.log(abc)-np.log(np.log(abc))+np.log(np.log(abc))/np.log(abc), lambertw( np.exp(abc), k=0).real )/b-a
        v = v_sgn * v_val
        v = np.clip(v, self.lower, self.upper)
        D=np.maximum(np.linalg.norm((self.x).flatten(), ord=1),np.linalg.norm((v).flatten(), ord=1))
        self.eta+=(eta_t/(D+1)*np.linalg.norm((self.x-v).flatten(), ord=1))**2
        eta_t_1=np.maximum(np.sqrt(self.eta),1)
        self.x[:]=(1.0-eta_t/eta_t_1)*self.x+eta_t/eta_t_1*v
        

def fmin(func, x0, upper,lower,l1=1.0,l2=1.0, func_p=None,stochastic=False, maxfev=50,batch=10,callback=None, epoch_size=10,eta=1.0):
    delta=np.sqrt(2*np.e *(2*np.log(x0.size-1)))/np.sqrt(maxfev)/x0.size
    b=batch
    if func_p is None:
        if stochastic:
            func_p=Grad_2p_Stoc(func,b,delta)
        else:
            func_p=Grad_2p_batch(func,b,delta)
    alg=AdaExpGradP(func=func,func_p=func_p,x0=x0,upper=upper,lower=lower,l1=l1,l2=l2,eta=eta)
    fev=1
    y=None
    while fev<=maxfev:
        y=alg.update()
        if stochastic:
            func.shuffle()
        if callback is not None and fev%epoch_size==0:
                res=OptimizeResult(func=func(y), x=y, nit=fev,
                          nfev=fev, success=(y is not None))
                callback(res)
        fev+=1
    return OptimizeResult(func=func(y), x=y, nit=fev,
                          nfev=fev, success=(y is not None))

class Grad_2p_batch:
    def __init__(self, func, n, delta):
        self.func=func
        self.b=n
        self.delta=delta
    def __call__(self, x):
        batch_v=np.random.choice([-1.0,1.0],size=x.shape,p=[0.5,0.5])
        batch=x+self.delta*batch_v
        for i in range(self.b-1):
            v = np.random.choice([-1.0,1.0],size=x.shape,p=[0.5,0.5])
            batch=np.append(batch,x+self.delta*v, axis=0) 
            batch_v=np.append(batch_v,v, axis=0)
        batch=np.append(batch, x, axis=0)   
        batch_y=self.func(batch)
        g=np.zeros(shape=x.shape)
        tilde_f_x_r= batch_y[self.b]
        for i in range(self.b):
            tilde_f_x_l= batch_y[i]
            g+=1.0/self.delta/self.b*(tilde_f_x_l-tilde_f_x_r)*batch_v[i]
        return g
        
class Grad_2p_Stoc:
    def __init__(self, func, n, delta):
        self.func=func
        self.b=n
        self.delta=delta
    def __call__(self, x):
        batch_v=np.random.choice([-1.0,1.0],size=x.shape,p=[0.5,0.5])
        batch_l=x+self.delta*batch_v
        batch_r=x
        for i in range(self.b-1):
            v = np.random.choice([-1.0,1.0],size=x.shape,p=[0.5,0.5])
            batch_l=np.append(batch_l,x+self.delta*v, axis=0) 
            batch_r=np.append(batch_r,x, axis=0) 
            batch_v=np.append(batch_v,v, axis=0)
        batch_y_l=self.func(batch_l)
        batch_y_r=self.func(batch_r)
        g=np.zeros(shape=x.shape)
        for i in range(self.b):
            tilde_f_x_l= batch_y_l[i]
            tilde_f_x_r= batch_y_r[i]
            g+=1.0/self.delta/self.b*(tilde_f_x_l-tilde_f_x_r)*batch_v[i]
        return g
        