from cmath import inf
from scipy.special import lambertw
from scipy.optimize import OptimizeResult
import numpy as np
import numpy.linalg as lina
class AccExp(object):
    def __init__(self, func, func_p, x0, lower, upper, l1=1.0, l2=1.0,eta=1.0):
        self.func = func
        self.func_p=func_p
        self.x= np.zeros(shape=x0.shape)
        self.y= np.zeros(shape=x0.shape)
        self.z= np.zeros(shape=x0.shape)
        self.x[:]= x0
        self.y[:]= x0
        self.z[:]= x0
        self.d = self.x.size
        self.lower=lower
        self.upper=upper
        self.lam=0.0
        self.l1=l1
        self.l2=l2
        self.eta=0.0
        self.D=1.0
        self.t=1.0
        self.tau=1.0
        self.ma=0.0
        self.va=0.0

    def update(self):
        g=self.func_p(x=self.z)
        self.mdy(g)
        self.mdx(g)
        self.t=1.0
        self.tau+=self.t
        gamma_t=self.t/self.tau
        self.z[:]=gamma_t*self.x+(1.0-gamma_t)*self.y
        return self.y
        
        
    def mdx(self,g):
        beta = 1.0 / self.d
        eta_t=np.maximum(1.0,(self.eta)**(1.0/2.0))
        z=(np.log(np.abs(self.x) / beta + 1.0)) * np.sign(self.x) -g*self.t/eta_t
        v_sgn = np.sign(z)
        if self.l2 == 0.0:
            v_val = beta * np.exp(np.maximum(np.abs(z) - self.l1*self.t/eta_t ,0.0)) - beta
        else:
            a = beta
            b = self.l2*self.t/eta_t
            c = np.minimum(self.l1*self.t/eta_t - np.abs(z),0.0)
            abc=-c+np.log(a*b)+a*b
            v_val = np.where(abc>=15.0,np.log(abc)-np.log(np.log(abc))+np.log(np.log(abc))/np.log(abc), lambertw( np.exp(abc), k=0).real )/b-a
        v = v_sgn * v_val
        self.x [:]= np.clip(v, self.lower, self.upper)
        D=np.maximum(lina.norm(self.x.flatten(),ord=1),lina.norm(v.flatten(),ord=1))
        self.eta+=((lina.norm((v-self.z).flatten(),ord=1)*eta_t/(1+D))**2)
        
    def mdy(self,g):
        beta = 1.0 / self.d
        eta_t=np.maximum(1.0,(self.eta)**(1.0/2.0))
        z=(np.log(np.abs(self.z) / beta + 1.0)) * np.sign(self.z) -g*self.t/eta_t
        v_sgn = np.sign(z)
        if self.l2 == 0.0:
            v_val = beta * np.exp(np.maximum(np.abs(z) - self.l1*self.t/eta_t ,0.0)) - beta
        else:
            a = beta
            b = self.l2*self.t/eta_t
            c = np.minimum(self.l1*self.t/eta_t - np.abs(z),0.0)
            abc=-c+np.log(a*b)+a*b
            v_val = np.where(abc>=15.0,np.log(abc)-np.log(np.log(abc))+np.log(np.log(abc))/np.log(abc), lambertw( np.exp(abc), k=0).real )/b-a
        v = v_sgn * v_val
        v = np.clip(v, self.lower, self.upper)
        D=np.maximum(lina.norm(self.x.flatten(),ord=1),lina.norm(v.flatten(),ord=1))
        self.eta+=((lina.norm((v-self.z).flatten(),ord=1)*eta_t/(1+D))**2)
        self.y[:]=v


def fmin(func,x0, upper,lower, l1=1.0,l2=1.0, maxfev=50,callback=None, epoch_size=10,eta=1.0):
    delta=1.0/maxfev/maxfev/x0.size
    func_p=Grad_2p_batch(func,maxfev,maxfev,delta)
    alg=AccExp(func=func,func_p=func_p,x0=x0,upper=upper,lower=lower,l1=l1,l2=l2,eta=1.0/maxfev)
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
    def __init__(self, func, n, q, delta):
        self.func=func
        self.n=n
        self.q=q
        self.delta=delta
    def __call__(self, x, y=None):
        if y is None:
            batch=np.zeros(shape=x.shape)
            batch_v=np.zeros(shape=x.shape)
            batch[:]=x
            batch_v[:]=x
            g=np.zeros(shape=x.shape)
            for i in range(self.n+1):
                v= np.random.choice([-1.0,1.0],size=x.shape,p=[0.5,0.5])
                batch=np.append(batch,x+self.delta*v, axis=0)
                batch_v=np.append(batch_v,v, axis=0)
            batch_y=self.func(batch)
            tilde_f_x_r= batch_y[0]
            for i in range(1,self.n+1):
                tilde_f_x_l= batch_y[i]
                g[0]+=1.0/self.delta/self.n*(tilde_f_x_l-tilde_f_x_r)*batch_v[i]
            return g
        else:
            batch_v=np.random.choice([-1.0,1.0],size=x.shape,p=[0.5,0.5])
            batch=x+self.delta*batch_v
            batch=np.append(batch,y+self.delta*batch_v, axis=0) 
            for i in range(self.q-1):
                v = np.random.choice([-1.0,1.0],size=x.shape,p=[0.5,0.5])
                batch=np.append(batch,x+self.delta*v, axis=0) 
                batch=np.append(batch,y+self.delta*v, axis=0) 
                batch_v=np.append(batch_v,v, axis=0)
            batch=np.append(batch, x, axis=0)   
            batch=np.append(batch, y, axis=0)     
            batch_y=self.func(batch)
            g=np.zeros(shape=x.shape)
            tilde_f_x_r= batch_y[2*self.q]
            tilde_f_y_r= batch_y[2*self.q+1]
            for i in range(self.q):
                tilde_f_x_l= batch_y[2*i]
                tilde_f_y_l= batch_y[2*i+1]
                g[0]+=1.0/self.delta/self.q*(tilde_f_x_l-tilde_f_x_r-tilde_f_y_l+tilde_f_y_r)*batch_v[i]
            return g