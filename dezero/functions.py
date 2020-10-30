import numpy as np
from .core import Function
from .core import as_variable
from dezero import utils 
from .core import Variable,as_array
from dezero import cuda, utils
import dezero


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
        gy = utils.reshape_sum_backward(gy,self.x_shape,self.axis,self.keepdims)
        gx = broadcast_to(gy, self.x_shape)
    
def sum(x, axis=None, keepdims=False):
    return Sum(axis,keepdims)(x)

class BroadcastTo(Function):
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
        #print(self.x_shape)
        y = utils.sum_to(x,self.shape)
        return y

    def backward(self,gy):
        gx = broadcast_to(gy, self.x_shape)

        return gx

def sum_to(x,shape):
    if x.shape == shape:
        return as_variable(x)

   
    return SumTo(shape)(x)

class MatMul(Function):
    def forward(self,x,W):
        y = x.dot(W)
        return y
    
    def backward(self,gy):
        x, W = self.inputs
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)

        return gx, gW
    
def matmul(x, W):
    return MatMul()(x,W)

class MeanSquaredError(Function):
    def forward(self,x0,x1):
        diff = x0 -x1
        y = (diff **2).sum() /len(diff)
        return y
    def backward(self,gy):
        x0,x1 = self.inputs
        diff = x0 -x1
        gy = broadcast_to(gy, diff.shape)
        gx0 = gy*diff*(2./ len(diff) )
        gx1 = -gx0

        return gx0, gx1

def mean_squared_error(x0,x1):
    return MeanSquaredError()(x0,x1)

class Exp(Function):
    def forward(self, x):
        #xp = cuda.get_array_module(x)
        y = np.exp(x)
        return y

    def backward(self, gy):
        y = self.outputs[0]()  # weakref
        gx = gy * y
        return gx


def exp(x):
    return Exp()(x)


def linear_simple(x,W,b=None):
    x, W = as_variable(x), as_variable(W)
    t = MatMul(x,W)
    if b is None:
        return t
    
    y = t + b
    t.data = None
    return y

def sigmoid_simple(x):
    x = as_variable(x)
    y = 1 / (1 + exp(-x))
    return y

class Linear(Function):
    def forward(self, x, W, b):
        y = x.dot(W)
        if b is not None:
            y += b
        return y

    def backward(self, gy):
        x, W, b = self.inputs
        gb = None if b.data is None else sum_to(gy, b.shape)
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW, gb

def linear(x, W, b=None):
    return Linear()(x, W, b)

class Sigmoid(Function):
    def forward(self, x):
        #xp = cuda.get_array_module(x)
        y = 1 / (1 + np.exp(-x))
        #y = tanh(x * 0.5) * 0.5 + 0.5  # Better implementation
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * y * (1 - y)
        return gx


def sigmoid(x):
    return Sigmoid()(x)

class GetItem(Function):
    def __init__(self, slices):
        self.slices = slices

    def forward(self, x):
        y = x[self.slices]
        return y

    def backward(self, gy):
        x, = self.inputs
        f = GetItemGrad(self.slices, x.shape)
        print(gy)
        return f(gy)


class GetItemGrad(Function):
    def __init__(self, slices, in_shape):
        self.slices = slices
        self.in_shape = in_shape

    def forward(self, gy):
        xp = dezero.cuda.get_array_module(gy)
        gx = xp.zeros(self.in_shape, dtype=gy.dtype)

        if xp is np:
            np.add.at(gx, self.slices, gy)
        else:
            xp.scatter_add(gx, self.slices, gy)
        return gx

    def backward(self, ggx):
        return get_item(ggx, self.slices)


def get_item(x, slices):
    f = GetItem(slices)
    return f(x)

class Log(Function):
    def forward(self, x):
        y = np.log(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy / x
        return gx


def log(x):
    return Log()(x)

def softmax_simple(x,axis=1):
    x = as_variable(x)
    y = exp(x)
    sum_y = sum(y, axis=axis, keepdims=True)
    return y / sum_y

def softmax_cross_entropy_simple(x,t):
    x,t = as_variable(x), as_variable(t)
    N = x.shape[0]

    p = softmax_simple(x)
    p = clip(p, 1e-15,1.0)
    log_p = log(p)
    tlog_p = log_p[np.arange(N), t.data]
    y = -1 * sum(tlog_p) / N

    return y

class Clip(Function):
    def __init__(self, x_min, x_max):
        self.x_min = x_min
        self.x_max = x_max

    def forward(self, x):
        y = np.clip(x, self.x_min, self.x_max)
        return y

    def backward(self, gy):
        x, = self.inputs
        mask = (x.data >= self.x_min) * (x.data <= self.x_max)
        gx = gy * mask
        return gx


def clip(x, x_min, x_max):
    return Clip(x_min, x_max)(x)

def accuracy(y, t):
    y,t = as_variable(y),as_variable(t)

    pred = y.data.argmax(axis=1).reshape(t.shape)
    result = (pred == t.data)
    acc = result.mean()
    return Variable(as_array(acc))

class SoftmaxCrossEntropy(Function):
    def forward(self, x, t):
        N = x.shape[0]
        log_z = utils.logsumexp(x, axis=1)
        log_p = x - log_z
        log_p = log_p[np.arange(N), t.ravel()]
        y = -log_p.sum() / np.float32(N)
        return y

    def backward(self, gy):
        x, t = self.inputs
        N, CLS_NUM = x.shape

        gy *= 1/N
        y = softmax(x)
        # convert to one-hot
        xp = cuda.get_array_module(t.data)
        t_onehot = xp.eye(CLS_NUM, dtype=t.dtype)[t.data]
        y = (y - t_onehot) * gy
        return y

def softmax_cross_entropy(x, t):
    return SoftmaxCrossEntropy()(x, t)

class Softmax(Function):
    def __init__(self, axis=1):
        self.axis = axis

    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = x - x.max(axis=self.axis, keepdims=True)
        y = xp.exp(y)
        y /= y.sum(axis=self.axis, keepdims=True)
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = y * gy
        sumdx = gx.sum(axis=self.axis, keepdims=True)
        gx -= y * sumdx
        return gx


def softmax(x, axis=1):
    return Softmax(axis)(x)

class ReLU(Function):
    def forward(self,x):
        y = np.maximum(x, 0.0)
        return y
    
    def backward(self,gy):
        x, = self.inputs
        mask = x.data > 0
        gx = gy*mask
        return gx

def relu(x):
    return ReLU()(x)

def dropout(x, dropout_ratio=0.5):
    x = as_variable(x)

    if dezero.Config.train:
        mask = np.random.rand(*x.shape) > dropout_ratio
        scale = np.array(1.0 - dropout_ratio).astype(x.dtype)
        y = x*mask / scale
        return y 
    
    else: 
        return x


from dezero.functions_conv import conv2d
from dezero.functions_conv import deconv2d
from dezero.functions_conv import conv2d_simple
from dezero.functions_conv import im2col
from dezero.functions_conv import col2im
from dezero.functions_conv import pooling_simple
from dezero.functions_conv import pooling
from dezero.functions_conv import average_pooling