from scipy.special import lambertw
from scipy.optimize import OptimizeResult
import numpy as np
import numpy.linalg as lina
class ExpStorm(object):
    def __init__(self, func, func_p, x0, lower, upper, l1=1.0, l2=1.0):
        self.func = func
        self.func_p=func_p
        self.x= np.zeros(shape=x0.shape)
        self.d_t= np.zeros(shape=x0.shape)
        self.x[:]= x0
        self.d = self.x.size
        self.lower=lower
        self.upper=upper
        self.l1=l1
        self.l2=l2
        self.eta=0.0
        self.t=0.0
        self.D=1.0
        self.x_prev= np.zeros(shape=x0.shape)

    def update(self):
        self.t+=1
        delta=1.0/self.t/self.d
        b= int(np.ceil(np.sqrt(self.t)))
        g,m=self.func_p(self.x,self.x_prev,b,delta)
        self.md(g,m)
        return self.x
    
        
    def md(self,g,m):
        self.x_prev[:]=self.x
        beta = 1.0 / self.d
        gamma=self.t**(-1.0/2.0)
        self.d_t=g+(1-gamma)*(self.d_t-m)
        eta_t=np.maximum((self.eta)**(1.0/2.0),1.0)
        z=(np.log(np.abs(self.x) / beta + 1.0)) * np.sign(self.x) - self.d_t/eta_t
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
        D=np.maximum(lina.norm(self.x.flatten(),ord=1),lina.norm(v.flatten(),ord=1))
        self.D=np.maximum(D,self.D)
        lam=1.0/(D+1.0)
        self.eta+=(((lina.norm((v-self.x).flatten(),ord=1)*eta_t)**2))
        eta_t_1=np.maximum((self.eta)**(1.0/2.0),1.0)
        self.x=(1.0-eta_t/eta_t_1)*self.x+eta_t/eta_t_1*v
        #self.x[:]=v
        

def fmin(func,x0, upper,lower,l1=1.0,l2=1.0, maxfev=50,callback=None, epoch_size=10,eta=1.0):
    func_p=Grad_2p_batch(func)
    alg=ExpStorm(func=func,func_p=func_p,x0=x0,upper=upper,lower=lower,l1=l1,l2=l2)
    fev=1
    y=None
    budget=0
    while budget<=maxfev*maxfev:
        y=alg.update()
        budget+=int(np.ceil(np.sqrt(fev)))*2
        if callback is not None and fev%epoch_size==0:
                res=OptimizeResult(func=func(y), x=y, nit=budget,
                          nfev=fev, success=(y is not None))
                callback(res)
        fev+=1
    
    return OptimizeResult(func=func(y), x=y, nit=budget,
                          nfev=fev, success=(y is not None))

class Grad_2p_batch:
    def __init__(self, func):
        self.func=func
    def __call__(self, x, y, n, delta):
        batch_v=np.random.choice([-1.0,1.0],size=x.shape,p=[0.5,0.5])
        batch=x+delta*batch_v
        batch=np.append(batch,y+delta*batch_v, axis=0) 
        for i in range(n-1):
            v = np.random.choice([-1.0,1.0],size=x.shape,p=[0.5,0.5])
            batch=np.append(batch,x+delta*v, axis=0) 
            batch=np.append(batch,y+delta*v, axis=0) 
            batch_v=np.append(batch_v,v, axis=0)
        batch=np.append(batch, x, axis=0)   
        batch=np.append(batch, y, axis=0)     
        batch_y=self.func(batch)
        m=np.zeros(shape=x.shape)
        g=np.zeros(shape=x.shape)
        tilde_f_x_r= batch_y[2*n]
        tilde_f_y_r= batch_y[2*n+1]
        for i in range(n):
            tilde_f_x_l= batch_y[2*i]
            tilde_f_y_l= batch_y[2*i+1]
            g+=1.0/delta/n*(tilde_f_x_l-tilde_f_x_r)*batch_v[i]
            m+=1.0/delta/n*(tilde_f_y_l-tilde_f_y_r)*batch_v[i]
        return g,m
