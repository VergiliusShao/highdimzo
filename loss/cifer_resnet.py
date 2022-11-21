from ssl import cert_time_to_seconds
from mxnet import context
from mxnet.gluon.model_zoo import vision
import mxnet as mx
import mxnet.ndarray as nd
from mxnet.gluon.model_zoo import vision
import numpy as np
from mxnet import autograd as ag

from mxnet import gluon
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms

from gluoncv.model_zoo import get_model
from gluoncv.utils import makedirs, TrainingHistory
from gluoncv.data import transforms as gcv_transforms
from numpy.core.fromnumeric import shape


class CEM:
    def __init__(self, index=2593, ctx=mx.cpu()):
        # Preparing the Data Samples
        self.ctx =ctx
        dataset= gluon.data.vision.CIFAR10(train=False).transform_first(transform_test)
        self.x=dataset[index][0].copyto(self.ctx).expand_dims(axis=0) 
        np_y=dataset[index][1]
        np_x=self.x.asnumpy()
        #self.ctx = [mx.cpu()]
        #initialising net
        self.net=net = get_model('cifar_resnet20_v1', classes=10,ctx=self.ctx)
        net.load_parameters('resnet20.params', ctx=self.ctx)
       
        mean = np.array([[0.4914, 0.4822, 0.4465]])
        std=np.array([[0.2023, 0.1994, 0.2010]]) 

        self.shape=np_x.shape
        self.origin=np.zeros(shape=self.shape)
        self.origin+=np_x
        
        self.pp_upper=np.zeros(shape=self.shape)
        self.pp_upper=np.maximum(np_x,0.0)
        self.pp_lower=np.zeros(shape=self.shape)
        self.pp_lower=np.minimum(np_x,0.0)
        self.pp_init=np.zeros(shape=self.shape)
        self.pp_init+=np_x
       
        self.pn_upper=((np.ones(shape=(1,3,32,32)).T-mean.T)/std.T).T
        self.pn_upper-=np_x
        self.pn_lower=((np.zeros(shape=(1,3,32,32)).T-mean.T)/std.T).T
        self.pn_lower-=np_x
        self.pn_init=np.zeros(shape=self.shape)

        self.aa_upper=((np.ones(shape=(1,3,32,32)).T-mean.T)/std.T).T
        self.aa_lower=-((np.ones(shape=(1,3,32,32)).T-mean.T)/std.T).T
        self.aa_init=np.zeros(shape=self.shape)
       
        self.y=self.net(self.x)
        self.label=nd.argmax(self.y)
        self.mask=mx.nd.array([1 for k in self.y[0]],ctx=self.ctx)
        self.mask[self.label]=0
        self.correct=(self.label==np_y)

        
    def pp(self,delta):
        mx_delta = mx.nd.array(delta.reshape(-1,3,32,32),ctx=self.ctx)
        y_delta_batch=(self.net(mx_delta))
        attack=[]
        for y_delta in y_delta_batch:
            #attack_value=mx.nd.maximum(mx.nd.max(mx.nd.contrib.boolean_mask(y_delta,self.mask))-y_delta[self.label],self.kappa).asnumpy()[0]
            attack_value=(mx.nd.max(mx.nd.contrib.boolean_mask(y_delta,self.mask))-y_delta[self.label]).asnumpy()[0]
            if attack_value<-10:
                loss=np.log(1.0+np.exp(attack_value))
            else:
                loss=attack_value+np.log(1.0+np.exp(-attack_value))
            attack.append(loss)
        if len(attack)==1:
            return attack[0]
        else:
            return np.array(attack)

    def pp_grad(self, delta):
        mx_delta = mx.nd.array(delta.reshape(-1,3,32,32),ctx=self.ctx)
        mx_delta.attach_grad()
        with ag.record():
            y_delta=(self.net(mx_delta))
            pos=mx.nd.maximum(mx.nd.max(mx.nd.contrib.boolean_mask(y_delta[0],self.mask))-y_delta[0][self.label],self.kappa)
        pos.backward()
        return mx_delta.grad.asnumpy()

    def pn(self,delta):
        mx_delta = mx.nd.array(delta.reshape(-1,3,32,32),ctx=self.ctx)
        y_delta_batch=(self.net(self.x+mx_delta))
        attack=[]
        for y_delta in y_delta_batch:
            attack_value=(y_delta[self.label]-mx.nd.max(mx.nd.contrib.boolean_mask(y_delta,self.mask))).asnumpy()[0]
            if attack_value<-10:
                loss=np.log(1.0+np.exp(attack_value))
            else:
                loss=attack_value+np.log(1.0+np.exp(-attack_value))
            attack.append(loss)
            #attack.append(mx.nd.maximum(y_delta[self.label]-mx.nd.max(mx.nd.contrib.boolean_mask(y_delta,self.mask)),self.kappa).asnumpy()[0])
        if len(attack)==1:
            return attack[0]
        else:
            return np.array(attack)

    def pn_grad(self, delta):
        mx_delta = mx.nd.array(delta.reshape(-1,3,32,32),ctx=self.ctx)
        mx_delta.attach_grad()
        with ag.record():
            y_delta=(self.net(mx_delta+self.x))
            pos=mx.nd.maximum(y_delta[0][self.label]-mx.nd.max(mx.nd.contrib.boolean_mask(y_delta[0],self.mask)),self.kappa)
        pos.backward()
        return mx_delta.grad.asnumpy()
    

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

class PP_Loss:
    def __init__(self, cem):
        self.cem=cem
    def __call__(self, delta):
        return self.cem.pp(delta)


class PN_Loss:
    def __init__(self, cem):
        self.cem=cem
    def __call__(self, delta):
        return self.cem.pn(delta)



class PP_Grad:
    def __init__(self, cem):
        self.cem=cem
    def __call__(self, delta):
        return self.cem.pp_grad(delta) 


class PN_Grad:
    def __init__(self, cem):
        self.cem=cem
    def __call__(self, delta):
        return self.cem.pn_grad(delta) 



class Grad_2p:
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
            v= np.random.normal(size=x.shape)
            v_norm= np.linalg.norm(v)
            v=v/v_norm
            batch=np.append(batch,x+self.delta*v, axis=0)
            batch_v=np.append(batch_v,v, axis=0)
        batch_y=self.func(batch)
        tilde_f_x_r= batch_y[0]
        for i in range(1,self.n+1):
            tilde_f_x_l= batch_y[i]
            g[0]+=d/self.delta/self.n*(tilde_f_x_l-tilde_f_x_r)*batch_v[i]
        return g

class Grad_2p_Rad:
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
            g[0]+=1.0/self.delta/self.n*(tilde_f_x_l-tilde_f_x_r)*batch_v[i]
        return g