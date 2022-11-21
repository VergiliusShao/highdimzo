from scipy.special import lambertw
from scipy.optimize import OptimizeResult
import numpy as np
import numpy.linalg as lina
class AOExpGrad(object):
    def __init__(self, func, func_p, x0, lower, upper, l1=1.0, l2=1.0,eta=1.0):
        self.func = func
        self.func_p=func_p
        self.x= np.zeros(shape=x0.shape)
        self.x[:]= x0
        self.y= np.zeros(shape=x0.shape)
        self.y[:]= x0
        self.d = self.x.size
        self.lower=lower
        self.upper=upper
        self.eta=eta
        self.lam=0.0
        self.t=0.0
        self.beta=0
        self.l1=l1
        self.l2=l2
        self.h=np.zeros(shape=self.x.shape)
        self.D=np.maximum(np.max(np.abs(self.upper)),np.max(np.abs(self.lower)))

    def update(self):
        self.t+=1.0
        self.beta+=self.t
        g=self.func_p(self.y)
        self.step(g)
        self.h[:]=g
        return self.y
        
    def step(self,g):
        self.update_parameters(g)
        self.md(g)

    def update_parameters(self,g):
        self.lam+=((self.t*lina.norm((g-self.h).flatten(),ord=np.inf))**2)
        
    def md(self,g):
        beta = 1.0 / self.d
        alpha = np.sqrt(self.lam)/np.sqrt(np.log(self.d*self.D+1))*self.eta
        if alpha==0.0:
            alpha+=1e-6
        z=(np.log(np.abs(self.x) / beta + 1.0)) * np.sign(self.x) - (self.t*g-self.t*self.h+(self.t+1)*g) / alpha
        x_sgn = np.sign(z)
        if self.l2 == 0.0:
            x_val = beta * np.exp(np.maximum(np.abs(z) - self.l1*(self.t+1) / alpha,0.0)) - beta
        else:
            a = beta
            b = self.l2*(self.t+1)/ alpha
            c = np.minimum(self.l1*(self.t+1) / alpha - np.abs(z),0.0)
            abc=-c+np.log(a*b)+a*b
            x_val = np.where(abc>=15.0,np.log(abc)-np.log(np.log(abc))+np.log(np.log(abc))/np.log(abc), lambertw( np.exp(abc), k=0).real )/b-a
        y = x_sgn * x_val
        self.x = np.clip(y, self.lower, self.upper)
        self.y=(self.t/self.beta)*self.x+((self.beta-self.t)/self.beta)*self.y

        
def fmin(func, x0, upper,lower,l1=1.0,l2=1.0, func_p=None,stochastic=False, maxfev=50,batch=10,callback=None, epoch_size=10,eta=1.0):
    delta=np.sqrt(2*np.e *(2*np.log(x0.size-1)))/np.sqrt(maxfev)/x0.size
    b=batch
    if func_p is None:
        if stochastic:
            func_p=Grad_2p_Stoc(func,b,delta)
        else:
            func_p=Grad_2p_batch(func,b,delta)
    alg=AOExpGrad(func=func,func_p=func_p,x0=x0,upper=upper,lower=lower,l1=l1,l2=l2,eta=eta)
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
        #batch_v=np.random.uniform(-1,1,size=x.shape)
        batch=x+self.delta*batch_v
        for i in range(self.b-1):
            v = np.random.choice([-1.0,1.0],size=x.shape,p=[0.5,0.5])
            #v = np.random.uniform(-1,1,size=x.shape)
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
        