from scipy.optimize import OptimizeResult
import numpy as np
import numpy.linalg as lina
class AdaPGD(object):
    def __init__(self, func, func_p, x0, lower, upper, l1=1.0, l2=1.0,eta=0.5):
        self.func = func
        self.func_p=func_p
        self.x= np.zeros(shape=x0.shape)
        self.x[:]= x0
        self.d = self.x.size
        self.lower=lower
        self.upper=upper
        self.eta=0.0
        self.l1=l1
        self.l2=l2
        self.p=eta

    def update(self):
        g=self.func_p(self.x)
        self.step(g)
        return self.x
                

    def step(self,g):
        self.md(g)

    def md(self,g):
        eta_t_m_1 = np.maximum((self.eta)**self.p,1.0)
        z = self.x-g/eta_t_m_1
        v_sgn = np.sign(z)
        v_val = np.maximum(eta_t_m_1*np.abs(z)-self.l1,0.0)/(eta_t_m_1+self.l2)
        v = v_sgn * v_val
        v = np.clip(v, self.lower, self.upper)
        self.eta+=((lina.norm((v-self.x).flatten(),ord=2)*eta_t_m_1)**2)
        eta_t = np.maximum((self.eta)**self.p,1.0)
        self.x=(1-eta_t_m_1/eta_t)*self.x+eta_t_m_1/eta_t*v


def fmin(func,x0, upper,lower,l1=1.0,l2=1.0, maxfev=50, batch=10, callback=None, epoch_size=10,eta=0.5):
    delta=1.0/np.sqrt(maxfev*x0.size)
    b=batch
    func_p=Grad_2p_batch(func,b,delta)
    alg=AdaPGD(func=func,func_p=func_p,x0=x0,upper=upper,lower=lower,l1=l1,l2=l2,eta=eta)
    fev=1
    y=None
    while fev <= maxfev:
        y=alg.update()
        if callback is not None and fev%epoch_size==0:
                res=OptimizeResult(func=func(y), x=y, nit=fev,
                          nfev=fev, success=(y is not None))
                callback(res)
        fev+=1
    return OptimizeResult(func=func(y), x=y, nit=fev,
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
        for i in range(1,self.n+1):
            tilde_f_x_l= batch_y[i]
            g+=1.0/self.delta/self.n*(tilde_f_x_l-tilde_f_x_r)*batch_v[i]
        return g
