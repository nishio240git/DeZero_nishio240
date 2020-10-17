import numpy as np
from .core import Function
from .core import as_variable
from dezero import utils 

class Tanh(Function):
    def forward(self,x):
        y = np.tanh(x)
        return y

    def backward(self,gy):
        y = self.outputs[0]()
        gx = gy*(1-y*y)
        return gx

def tanh(x):
    return Tanh()(x)

class Reshape(Function):
    def __init__(self,shape):
        self.shape = shape
    
    def forward(self,x):
        self.x_shape = x.shape
        y = x.reshape(self.shape)

        return y
    
    def backward(self,gy):
        return reshape(gy,self.x_shape)
    
def reshape(x,shape):
    if x.shape == shape:
        return as_variable(x)
    
    return Reshape(shape)(x)


class Transpose(Function):
    def forward(self,x):
        y = np.transpose(x)
        return y
    
    def backward(self,gy):
        gx = transpose(gy)
        return gx
    
def transpose(x):
    return Transpose()(x)

class Sum(Function):
    def __init__(self,axis,keepdims):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self,x):
        self.x_shape = x.shape
        y = x.sum(axis=self.axis, keepdims=self.keepdims)

        return y
    
    def backward(self,gy):
        gy = utils.rehsape_sum_backward(gy,self.x_shape,self.axis,self.keepdims)
        gx = broadcast_to(gy, self.x_shape)
    
def sum(x, axis=None, keepdims=False):
    return Sum(axis,keepdims)(x)

def BroadcastTo(Function):
    def __init__(self,shape):
        self.shape = shape
    
    def forward(self,x):
        self.x_shape = x.shape
        y = np.broadcast_to(x,self.shape)
        return y 

    def backward(self,gy):
        gx = sum_to(gy,self.x_shape)
        return gx
    
def broadcast_to(x,shape):
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)

class SumTo(Function):
    def __init__(self,shape):
        self.shape = shape
    def forward(self,x):
        self.x_shape = x.shape
        y = utils.sum_to(x,self.x_shape)
        return y

    def backward(self,gy):
        gx = broadcast_to(gy, self.x_shape)

        return gx

def sum_to(x,shape):
    if x.shape == shape:
        return as_variable(x)

    return SumTo(shape)(x)



