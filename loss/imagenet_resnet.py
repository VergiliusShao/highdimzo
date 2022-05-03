from mxnet import context
from mxnet.gluon.model_zoo import vision
import mxnet as mx
import mxnet.ndarray as nd
from mxnet.gluon.model_zoo import vision
import numpy as np
from mxnet import autograd as ag
import mxnet.image as image
from mxnet import gluon
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms
import os
from gluoncv.data import ImageNet1kAttr
from gluoncv.data.transforms.presets.imagenet import transform_eval
from gluoncv.model_zoo import get_model
from gluoncv.utils import makedirs, TrainingHistory
from gluoncv.data import transforms as gcv_transforms
from numpy.core.fromnumeric import shape
from gluoncv.data import ImageNet
from mxnet.gluon.data.vision import transforms
from mxnet.gluon.data import DataLoader
IMG_SIZE=224

class CEM:
    def __init__(self, folder_path,index=2595,kappa=2,ctx=mx.cpu()):
        # Preparing the Data Samples
        self.ctx =ctx
        val_data=ImageNet(root=folder_path,train=False)
        
        self.x=transform_eval(val_data[index][0]).as_in_context(ctx)
        np_y=val_data[index][1]
        np_x=self.x.asnumpy().reshape(1,IMG_SIZE*3,IMG_SIZE)
        self.kappa=mx.nd.array([-kappa],ctx=self.ctx)
        
        #initialising net
        self.net=get_model('ResNet50_v2', pretrained=True,ctx=self.ctx)
        self.classes=self.net.classes
        std=np.array([[0.229, 0.224, 0.225]]) 

        self.shape=np_x.shape
        self.origin=np.zeros(shape=self.shape)
        self.origin+=np_x
        
        self.pp_upper=np.zeros(shape=self.shape)
        self.pp_upper=np.maximum(np_x,0.0)
        self.pp_lower=np.zeros(shape=self.shape)
        self.pp_lower=np.minimum(np_x,0.0)
        self.pp_init=np.zeros(shape=self.shape)
        self.pp_init+=np_x
        
        self.pn_upper=(np.ones(shape=(1,3,IMG_SIZE,IMG_SIZE)).T/std.T).T.reshape(1,IMG_SIZE*3,IMG_SIZE)
        self.pn_upper-=np_x
        self.pn_lower=(-np.ones(shape=(1,3,IMG_SIZE,IMG_SIZE)).T/std.T).T.reshape(1,IMG_SIZE*3,IMG_SIZE)
        self.pn_lower-=np_x
        self.pn_init=np.zeros(shape=self.shape)
        #self.pn_init=(self.pn_upper+self.pn_lower)/2
        self.y=self.net(self.x)
        self.label=nd.argmax(self.y)
        self.mask=mx.nd.array([1 for k in self.y[0]],ctx=self.ctx)
        self.mask[self.label]=0
        self.correct=(self.label==np_y)

        
    def pp(self,delta):
        mx_delta = mx.nd.array(delta.reshape(-1,3,IMG_SIZE,IMG_SIZE),ctx=self.ctx)
        y_delta_batch=(self.net(mx_delta))
        attack=[]
        for y_delta in y_delta_batch:
            attack.append(mx.nd.maximum(mx.nd.max(mx.nd.contrib.boolean_mask(y_delta,self.mask))-y_delta[self.label],self.kappa).asnumpy()[0])
        if len(attack)==1:
            return attack[0]
        else:
            return np.array(attack)
        #return mx.nd.maximum(mx.nd.max(mx.nd.contrib.boolean_mask(y_delta[0],self.mask))-y_delta[0][self.label],self.kappa).asnumpy()[0]

    def pp_grad(self, delta):
        mx_delta = mx.nd.array(delta.reshape(-1,3,IMG_SIZE,IMG_SIZE),ctx=self.ctx)
        mx_delta.attach_grad()
        with ag.record():
            y_delta=(self.net(mx_delta))
            pos=mx.nd.maximum(mx.nd.max(mx.nd.contrib.boolean_mask(y_delta[0],self.mask))-y_delta[0][self.label],self.kappa)
        pos.backward()
        return mx_delta.grad.asnumpy().reshape(1,IMG_SIZE*3,IMG_SIZE)

    def pn(self,delta):
        mx_delta = mx.nd.array(delta.reshape(-1,3,IMG_SIZE,IMG_SIZE),ctx=self.ctx)
        y_delta_batch=(self.net(self.x+mx_delta))
        attack=[]
        for y_delta in y_delta_batch:
            attack.append(mx.nd.maximum(y_delta[self.label]-mx.nd.max(mx.nd.contrib.boolean_mask(y_delta,self.mask)),self.kappa).asnumpy()[0])
        if len(attack)==1:
            return attack[0]
        else:
            return np.array(attack)
        #mx_delta = mx.nd.array(delta,ctx=self.ctx)
        #y_delta=(self.net(mx_delta+self.x))
        #return mx.nd.maximum(y_delta[0][self.label]-mx.nd.max(mx.nd.contrib.boolean_mask(y_delta[0],self.mask)),self.kappa).asnumpy()[0]

    def pn_grad(self, delta):
        mx_delta = mx.nd.array(delta.reshape(-1,3,IMG_SIZE,IMG_SIZE),ctx=self.ctx)
        mx_delta.attach_grad()
        with ag.record():
            y_delta=(self.net(mx_delta+self.x))
            pos=mx.nd.maximum(y_delta[0][self.label]-mx.nd.max(mx.nd.contrib.boolean_mask(y_delta[0],self.mask)),self.kappa)
        pos.backward()
        return mx_delta.grad.asnumpy().reshape(1,3*IMG_SIZE,IMG_SIZE)
    
class PP_Loss:
    def __init__(self, cem):
        self.cem=cem
    def __call__(self, delta):
        return self.cem.pp(delta)
class PP_Grad:
    def __init__(self, cem):
        self.cem=cem
    def __call__(self, delta):
        return self.cem.pp_grad(delta) 


class PN_Loss:
    def __init__(self, cem):
        self.cem=cem
    def __call__(self, delta):
        return self.cem.pn(delta)
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
        for i in range(1,self.n):
            tilde_f_x_l= batch_y[i]
            g[0]+=d/self.delta/self.n*(tilde_f_x_l-tilde_f_x_r)*batch_v[i]
        return g
