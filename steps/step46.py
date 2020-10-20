if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dezero import Variable,Model
import dezero.functions as F
from dezero import optimizers
import numpy as np
import dezero.layers as L 
from dezero.models import MLP

np.random.seed(0)
x = np.random.rand(100,1)
y = np.sin(2 *np.pi *x) + np.random.rand(100,1)


lr = 0.2
max_iter = 10000
hidden_size = 10

#モデルの定義
model = MLP((hidden_size,1))
optimizer = optimizers.SGD(lr)
optimizer.setup(model)


#学習の開始
for i in range(max_iter):
    y_pred = model(x)
    loss = F.mean_squared_error(y,y_pred)

    model.cleargrads()
    loss.backward()

    optimizer.update()

    for p in model.params():
        p.data -= lr * p.grad.data
    if i % 1000 == 0:
        print(loss)
