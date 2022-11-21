from scipy.special import lambertw
from scipy.optimize import OptimizeResult
import numpy as np
import numpy.linalg as lina
class ExpVR(object):
    def __init__(self, func, func_p, x0, lower, upper, l1=1.0, l2=1.0,p=0.5,batch=1.0):
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
        self.eta=1.0
        self.t=0.0
        self.b=batch
        self.x_prev= np.zeros(shape=x0.shape)
        self.gamma=1.0
        #self.L=1.0

    def update(self):
        self.t+=1.0
        if self.t==1.0:
            g,_=self.func_p(self.x,None)
            self.md(g,None)    
        else:
            g,m=self.func_p(self.x,self.x_prev)
            self.md(g,m)    
        return self.x
        
    def md(self,g,m):
        beta = 1.0 / self.d
        tau=(self.gamma)**(2.0/3.0)
        lr=np.maximum(1.0,(tau-1)/np.sqrt(tau))
        gamma=2.0/(tau+1)
        if m is None:
            self.d_t[:]=g
        else:
            self.d_t[:]=g+(1-gamma)*(self.d_t-m)
        eta_t=np.sqrt(self.eta)*lr        
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
        D=np.maximum(np.linalg.norm((self.x).flatten(), ord=1),np.linalg.norm((v).flatten(), ord=1))
        self.gamma+=1.0/self.b
        self.eta+=((eta_t/(1.0+D)*np.linalg.norm((self.x-v).flatten(), ord=1))**2)
        tau=(self.gamma)**(2.0/3.0)
        lr=np.maximum(1.0,(tau-1)/np.sqrt(tau))
        eta_t_1=np.sqrt(self.eta)*lr
        self.x_prev[:]=self.x
        self.x[:]=(1.0-eta_t/eta_t_1)*self.x+eta_t/eta_t_1*v

        
def fmin(func,x0, upper,lower,l1=1.0,l2=1.0,func_p=None,stochastic=False, maxfev=50, batch =10,callback=None, epoch_size=10,eta=0.5):
    delta=1.0/((maxfev)**(2.0/3.0))/x0.size
    n=int(batch)
    b=int(batch)
    if func_p is None:
        if stochastic:
            func_p=Grad_2p_Stoc(func,b,n,delta)
        else:
            func_p=Grad_2p_batch(func,b,n,delta)
    alg=ExpVR(func=func,func_p=func_p,x0=x0,upper=upper,lower=lower,l1=l1,l2=l2,batch=b,p=eta)
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
    def __init__(self, func, n, b, delta):
        self.func=func
        self.delta=delta
        self.n=n
        self.b=b
    def __call__(self, x, y=None):
        if y is None:
            batch_v=np.random.choice([-1.0,1.0],size=x.shape,p=[0.5,0.5])
            batch=x+self.delta*batch_v
            for i in range(self.b-1):
                v = np.random.choice([-1.0,1.0],size=x.shape,p=[0.5,0.5])
                batch=np.append(batch,x+self.delta*v, axis=0) 
                batch_v=np.append(batch_v,v, axis=0)    
            batch=np.append(batch,x, axis=0) 
            batch_y=self.func(batch)
            g=np.zeros(shape=x.shape)
            tilde_f_x_r= batch_y[self.b]
            for i in range(self.b):
                tilde_f_x_l= batch_y[i]
                g+=1.0/self.delta/self.b*(tilde_f_x_l-tilde_f_x_r)*batch_v[i]
            return g,None
        else:
            batch_v=np.random.choice([-1.0,1.0],size=x.shape,p=[0.5,0.5])
            batch=x+self.delta*batch_v
            batch=np.append(batch,y+self.delta*batch_v, axis=0)
            for i in range(self.n-1):
                v = np.random.choice([-1.0,1.0],size=x.shape,p=[0.5,0.5])
                batch=np.append(batch,x+self.delta*v, axis=0) 
                batch=np.append(batch,y+self.delta*v, axis=0) 
                batch_v=np.append(batch_v,v, axis=0)
            batch=np.append(batch,x, axis=0) 
            batch=np.append(batch,y, axis=0) 
            batch_y=self.func(batch)
            m=np.zeros(shape=x.shape)
            g=np.zeros(shape=x.shape)
            tilde_f_x_r= batch_y[2*self.n]
            tilde_f_y_r= batch_y[2*self.n+1]
            for i in range(self.n):
                tilde_f_x_l= batch_y[2*i]
                tilde_f_y_l= batch_y[2*i+1]
                g+=(1.0/self.delta/self.n*(tilde_f_x_l-tilde_f_x_r)*batch_v[i])
                m+=(1.0/self.delta/self.n*(tilde_f_y_l-tilde_f_y_r)*batch_v[i])
            return g,m

class Grad_2p_Stoc:
    def __init__(self, func, n, b, delta):
        self.func=func
        self.delta=delta
        self.n=n
        self.b=b
    def __call__(self, x, y=None):
        if y is None:
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
            return g,None
        else:
            batch_v=np.random.choice([-1.0,1.0],size=x.shape,p=[0.5,0.5])
            batch_l_x=x+self.delta*batch_v
            batch_l_y=y+self.delta*batch_v
            batch_r_x=x
            batch_r_y=y
            for i in range(self.n-1):
                v = np.random.choice([-1.0,1.0],size=x.shape,p=[0.5,0.5])
                batch_l_x=np.append(batch_l_x,x+self.delta*v, axis=0) 
                batch_l_y=np.append(batch_l_y,y+self.delta*v, axis=0) 
                batch_r_x=np.append(batch_r_x,x, axis=0) 
                batch_r_y=np.append(batch_r_y,y, axis=0) 
                batch_v=np.append(batch_v,v, axis=0)
            value_l_x=self.func(batch_l_x)
            value_l_y=self.func(batch_l_y)
            value_r_x=self.func(batch_r_x)
            value_r_y=self.func(batch_r_y)
            m=np.zeros(shape=x.shape)
            g=np.zeros(shape=x.shape)
            for i in range(self.n):
                tilde_f_x_l= value_l_x[i]
                tilde_f_y_l= value_l_y[i]
                tilde_f_x_r= value_r_x[i]
                tilde_f_y_r= value_r_y[i]
                g+=1.0/self.delta/self.n*(tilde_f_x_l-tilde_f_x_r)*batch_v[i]
                m+=1.0/self.delta/self.n*(tilde_f_y_l-tilde_f_y_r)*batch_v[i]
            return g,m
