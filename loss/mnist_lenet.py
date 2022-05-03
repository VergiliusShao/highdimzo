from __future__ import print_function
import mxnet as mx
from mxnet.ndarray import ndarray
import numpy as np
from mxnet import nd
import mxnet.ndarray as F

from mxnet import gluon
from mxnet.gluon import nn
from mxnet import autograd as ag

# Fixing the random seed
mx.random.seed(42)


class Net(gluon.Block):
    def __init__(self, **kwargs):
        super(Net, self).__init__(**kwargs)
        with self.name_scope():
            # layers created in name_scope will inherit name space
            # from parent layer.
            self.conv1 = nn.Conv2D(20, kernel_size=(5,5))
            self.pool1 = nn.MaxPool2D(pool_size=(2,2), strides = (2,2))
            self.conv2 = nn.Conv2D(50, kernel_size=(5,5))
            self.pool2 = nn.MaxPool2D(pool_size=(2,2), strides = (2,2))
            self.fc1 = nn.Dense(500)
            self.fc2 = nn.Dense(10)

    def forward(self, x):
        x = self.pool1(F.tanh(self.conv1(x)))
        x = self.pool2(F.tanh(self.conv2(x)))
        # 0 means copy over size from corresponding dimension.
        # -1 means infer size from the rest of dimensions.
        x = x.reshape((0, -1))
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        return x

class CEM:
    def __init__(self, index=2593,kappa=2,ctx=mx.cpu()):
    # Preparing the dataset:
        self.ctx=ctx
        self.kappa=mx.nd.array([-kappa],ctx=self.ctx)
        mnist = mx.test_utils.get_mnist()
        batch_size = 100
        model_file = "LeNet on MNist.model"
        self.val_data = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size)
        np_x=mnist['test_data'][index:index+1]
        np_y=mnist['test_label'][index:index+1]
        self.shape=np_x.shape
        self.origin=np.zeros(shape=self.shape)
        self.origin+=np_x

        self.pp_upper=np.zeros(shape=self.shape)
        self.pp_upper+=np_x
        self.pp_lower=np.zeros(shape=self.shape)
        self.pp_init=np.zeros(shape=self.shape)
        self.pp_init+=np_x
        #self.pp_init=(self.pp_lower+self.pp_upper)/2

        self.pn_upper=np.ones(shape=self.shape)
        self.pn_upper-=np_x
        self.pn_lower=np.zeros(shape=self.shape)
        self.pn_lower-=np_x
        self.pn_init=(self.pn_lower+self.pn_upper)/2
        #self.pn_init=np.zeros(shape=self.shape)
        #self.pn_init+=np_x
        self.x=nd.array(np_x, ctx = self.ctx)
        self.net=Net() 
        try:
            self.net.load_parameters(model_file, ctx = self.ctx)
        except:
            self.train_data = mx.io.NDArrayIter(mnist['train_data'], mnist['train_label'], batch_size, shuffle=True)
            self.net.initialize(mx.init.Xavier(magnitude=2.24), ctx=self.ctx)
            self.train()
            self.net.save_parameters(filename = model_file)
        finally:
            self.x.attach_grad()
            self.y=self.net(self.x)
            self.label=nd.argmax(self.y)
            self.mask=mx.nd.array([1 for k in self.y[0]],ctx=self.ctx)
            self.mask[self.label]=0
            self.correct=(np_y[0]==self.label[0])

    '''
    #Validate (Optional)
    net.validate(val_data, ctx)
    '''
    def pp(self,delta):        
        mx_delta = mx.nd.array(delta,ctx=self.ctx)
        y_delta_batch=(self.net(mx_delta))
        attack=[]
        for y_delta in y_delta_batch:
            attack.append(mx.nd.maximum(mx.nd.max(mx.nd.contrib.boolean_mask(y_delta,self.mask))-y_delta[self.label],self.kappa).asnumpy()[0])
        if len(attack)==1:
            return attack[0]
        else:
            return np.array(attack)

    def pp_grad(self, delta):
        mx_delta = mx.nd.array(delta)
        mx_delta.attach_grad()
        with ag.record():
            #y_delta=nd.softmax(self.net(mx_delta))
            y_delta=(self.net(mx_delta))
            pos=mx.nd.maximum(mx.nd.max(mx.nd.contrib.boolean_mask(y_delta[0],self.mask))-y_delta[0][self.label],self.kappa)
        pos.backward()
        return mx_delta.grad.asnumpy()
    
    def pn(self,delta):
        mx_delta = mx.nd.array(delta,ctx=self.ctx)
        #mx_delta = mx.nd.array(delta.reshape(-1,3,32,32),ctx=self.ctx)
        y_delta_batch=(self.net(self.x+mx_delta))
        attack=[]
        for y_delta in y_delta_batch:
            attack.append(mx.nd.maximum(y_delta[self.label]-mx.nd.max(mx.nd.contrib.boolean_mask(y_delta,self.mask)),self.kappa).asnumpy()[0])
        if len(attack)==1:
            return attack[0]
        else:
            return np.array(attack)

    def pn_grad(self, delta):
        mx_delta = mx.nd.array(delta)
        mx_delta.attach_grad()
        with ag.record():
            #y_delta=nd.softmax(self.net(mx_delta+self.x))
            y_delta=(self.net(mx_delta+self.x))
            pos=mx.nd.maximum(y_delta[0][self.label]-mx.nd.max(mx.nd.contrib.boolean_mask(y_delta[0],self.mask)),self.kappa)
        pos.backward()
        return mx_delta.grad.asnumpy()

    def predict(self,delta):
        mx_delta = mx.nd.array(delta)
        y_delta=nd.softmax(self.net(mx_delta))
        return nd.argmax(y_delta)


    def train(self):
        epoch = 10
        # Use Accuracy as the evaluation metric.
        trainer = gluon.Trainer(self.net.collect_params(), 'sgd', {'learning_rate': 0.02})
        metric = mx.metric.Accuracy()
        softmax_cross_entropy_loss = gluon.loss.SoftmaxCrossEntropyLoss()
        for i in range(epoch):
            # Reset the train data iterator.
            self.train_data.reset()
            # Loop over the train data iterator.
            for batch in self.train_data:
                # Splits train data into multiple slices along batch_axis
                # and copy each slice into a context.
                data = gluon.utils.split_and_load(batch.data[0], ctx_list=self.ctx, batch_axis=0)
                # Splits train labels into multiple slices along batch_axis
                # and copy each slice into a context.
                label = gluon.utils.split_and_load(batch.label[0], ctx_list=self.ctx, batch_axis=0)
                outputs = []
                # Inside training scope
                with ag.record():
                    for x, y in zip(data, label):
                        z = self.net(x)
                        # Computes softmax cross entropy loss.
                        loss = softmax_cross_entropy_loss(z, y)
                        # Backpropagate the error for one iteration.
                        loss.backward()
                        outputs.append(z)
                # Updates internal evaluation
                metric.update(label, outputs)
                # Make one step of parameter update. Trainer needs to know the
                # batch size of data to normalize the gradient by 1/batch_size.
                trainer.step(batch.data[0].shape[0])
            # Gets the evaluation result.
            name, acc = metric.get()
            # Reset evaluation result to initial state.
            metric.reset()
            print('training acc at epoch %d: %s=%f'%(i, name, acc))
        metric = mx.metric.Accuracy()
        # Reset the validation data iterator.
        self.val_data.reset()
        # Loop over the validation data iterator.
        for batch in self.val_data:
            # Splits validation data into multiple slices along batch_axis
            # and copy each slice into a context.
            data = gluon.utils.split_and_load(batch.data[0], ctx_list=self.ctx, batch_axis=0)
            # Splits validation label into multiple slices along batch_axis
            # and copy each slice into a context.
            label = gluon.utils.split_and_load(batch.label[0], ctx_list=self.ctx, batch_axis=0)
            outputs = []
            for x in data:
                outputs.append(self.net(x))
        # Updates internal evaluation
        metric.update(label, outputs)
        print('validation acc: %s=%f'%metric.get())

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