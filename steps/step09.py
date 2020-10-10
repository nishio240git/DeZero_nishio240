import numpy as np


class Variable:
    def __init__(self,data):
        
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))

        self.data = data
        self.grad  = None
        self.creator = None

    def set_creator(self,func):
        self.creator = func

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]
        while funcs:
            f = funcs.pop() #関数を取得
            x, y = f.input, f.output #関数の入出力を取得
            x.grad = f.backward(y.grad) #backwardメソッドを呼ぶ

            if x.creator is not None:
                funcs.append(x.creator) #一つ前の関数をリストに追加
        

class Function:
    def __call__(self,input):
        x = input.data
        y = self.forward(x)
        output = Variable(as_array(y))
        output.set_creator(self)
        self.input = input
        self.output = output
        return output

    def forward(self, x):
        raise NotImplementedError

    def backward(self,gy):
        raise NotImplementedError


class Square(Function):
    def forward(self,x):
        y = x **2
        return y

    def backward(self,gy):
        x = self.input.data
        gx = 2*x*gy
        return gx

class Exp(Function):
    def forward(self,x):
        y = np.exp(x)
        return y
    
    def backward(self,gy):
        x = self.input.data
        gx = np.exp(x)*gy
        return gx


def square(x):
    return Square()(x)

def exp(x):
    return Exp()(x)

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x



x = Variable(np.array(0.5))
y = square(exp(square(x))) ##連続して適用

"""
y.grad = np.array(1.0)
b.grad = C.backward(y.grad)
a.grad = B.backward(b.grad)
x.grad = A.backward(a.grad)

print(x.grad)"
"""


#y.grad = np.array(1.0)
y.backward()
print(x.grad)