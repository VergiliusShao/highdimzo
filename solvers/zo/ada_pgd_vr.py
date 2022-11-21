
from scipy.optimize import OptimizeResult
import numpy as np
import numpy.linalg as lina
class PGDVR(object):
    def __init__(self, func, func_p, x0, lower, upper, batch=1, l1=1.0, l2=1.0,p=1.0,k=1.0):
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
        self.t=1.0
        self.b=batch
        self.gamma=p
        self.alpha=1.0
        self.x_prev= np.zeros(shape=x0.shape)
        self.k=k
        self.c=2.0/(3.0*(self.k**3.0))+5.0/4.0
        self.m=np.maximum(self.k,(self.c*self.k)**3)     
        self.m=np.maximum(2.0,self.m)

    def update(self):
        if self.t==1.0:
            g,_=self.func_p(self.x)
            self.check(g)
        else:
            g,m=self.func_p(self.x,self.x_prev)
            self.md(g,m)
        self.t+=1.0
        return self.x
    
    def check(self,g):
        self.x_prev[:]=self.x
        self.d_t=g
        eta_t=self.k/((self.t+self.m)**(1.0/3.0))
        z = self.x-self.d_t*self.gamma
        v_sgn = np.sign(z)
        v_val = np.maximum(np.abs(z)/self.gamma-self.l1,0.0)/(1.0/self.gamma+self.l2)
        v = v_sgn * v_val
        v = np.clip(v, self.lower, self.upper)
        self.x[:]=self.x+(v-self.x)*eta_t
        self.alpha=np.minimum(self.c*eta_t*eta_t,1.0)
        #self.x[:]=v

        
    def md(self,g,m):
        self.x_prev[:]=self.x
        eta_t=self.k/((self.t+self.m)**(1.0/3.0))
        self.d_t=g+(1-self.alpha)*(self.d_t-m)
        z = self.x-self.d_t*self.gamma
        v_sgn = np.sign(z)
        v_val = np.maximum(np.abs(z)/self.gamma-self.l1,0.0)/(1.0/self.gamma+self.l2)
        v = v_sgn * v_val
        v = np.clip(v, self.lower, self.upper)
        self.x[:]=self.x+(v-self.x)*eta_t
        self.alpha=np.minimum(self.c*eta_t*eta_t,1.0)

def fmin(func,x0, upper,lower,l1=1.0,l2=1.0, maxfev=50,batch=10,callback=None, epoch_size=10,eta=0.5):
    b=int(batch)
    k=1.0
    c=2.0/(3.0*(k**3.0))+5.0/4.0
    m=np.maximum(k,(c*k)**3)     
    m=np.maximum(2.0,m)
    delta=1.0/((m+maxfev)**(2.0/3.0))/x0.size
    func_p=Grad_2p_batch(func,b,batch,delta)
    alg=PGDVR(func=func,func_p=func_p,x0=x0,upper=upper,lower=lower,batch=b,l1=l1,l2=l2,p=eta,k=k)
    fev=1
    y=None
    while fev<=maxfev:
        y=alg.update()
        if callback is not None and fev%epoch_size==0:
                res=OptimizeResult(func=func(y), x=y, nit=fev,
                          nfev=fev, success=(y is not None))
                callback(res)
        fev+=1
    
    return OptimizeResult(func=func(y), x=y, nit=fev,
                          nfev=fev, success=(y is not None))

class Grad_2p_batch:
    def __init__(self, func,n,b,delta):
        self.func=func
        self.delta=delta
        self.n=n
        self.b=b
    def __call__(self, x, y=None):
        d=x.size
        if y is None:
            batch_v=np.random.normal(size=x.shape)
            batch_v=batch_v/np.linalg.norm(batch_v.flatten(),ord=2)
            batch=x+self.delta*batch_v
            for i in range(self.b-1):
                v =np.random.normal(size=x.shape)
                v=v/np.linalg.norm(v.flatten(),ord=2)
                batch=np.append(batch,x+self.delta*v, axis=0) 
                batch_v=np.append(batch_v,v, axis=0)
            batch=np.append(batch, x, axis=0)   
            batch_y=self.func(batch)
            g=np.zeros(shape=x.shape)
            tilde_f_x_r= batch_y[self.b]
            for i in range(self.b):
                tilde_f_x_l= batch_y[i]
                g+=d/self.delta/self.b*(tilde_f_x_l-tilde_f_x_r)*batch_v[i]
            return g,None
        else:
            batch_v=np.random.normal(size=x.shape)
            batch_v=batch_v/np.linalg.norm(batch_v.flatten(),ord=2)
            batch=x+self.delta*batch_v
            batch=np.append(batch,y+self.delta*batch_v, axis=0) 
            for i in range(self.n-1):
                v =np.random.normal(size=x.shape)
                v=v/np.linalg.norm(v.flatten(),ord=2)
                batch=np.append(batch,x+self.delta*v, axis=0) 
                batch=np.append(batch,y+self.delta*v, axis=0) 
                batch_v=np.append(batch_v,v, axis=0)
            batch=np.append(batch, x, axis=0)   
            batch=np.append(batch, y, axis=0)     
            batch_y=self.func(batch)
            g=np.zeros(shape=x.shape)
            m=np.zeros(shape=x.shape)
            tilde_f_x_r= batch_y[2*self.n]
            tilde_f_y_r= batch_y[2*self.n+1]
            for i in range(self.n):
                tilde_f_x_l= batch_y[2*i]
                tilde_f_y_l= batch_y[2*i+1]
                g+=d/self.delta/self.n*(tilde_f_x_l-tilde_f_x_r)*batch_v[i]
                m+=d/self.delta/self.n*(tilde_f_y_l-tilde_f_y_r)*batch_v[i]
            return g,m
