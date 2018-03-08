import numpy as np
import utils.Variables as uvar
from utils.layers import svm_loss
from utils.optim import adam


class NNModel():
    def __init__(self):
        self.fc0 = AfineLayer(28 * 28, 369)
        self.relu0 = ReLULayer()
        self.fc1 = AfineLayer(369, 10)
        # self.relu1 = ReLULayer()

    def forward(self, x):
        out = self.fc0.forward(x)
        out = self.relu0.forward(out)
        out = self.fc1.forward(out)
        # out = self.relu1.forward(out)
        return out

    def backward(self, dout):
        # dout = self.relu1.backward(dout)
        dout = self.fc1.backward(dout)
        dout = self.relu0.backward(dout)
        dout = self.fc0.backward(dout)
        return dout

    def optim(self, learning_rate):
        self.fc0.optim(learning_rate)
        self.fc1.optim(learning_rate)

    def train(self, xs, ys, epochs, batchs=100, learning_rate=1e-2, display_per_epoch=10):
        """
        - xs: numpy array, shape of (N, W, H)
        - ys: numpy array, shape of (N,)
        where N is the number of datas
        W is the width of image
        H is the height of image
        - epochs: train 
        - batchs: N/batchs = k, (k \in N)
        """
        N, = ys.shape
        assert N % batchs == 0, "N/batchs (%d) is not a integer" % (N/batchs)
        C = N // batchs
        xs = np.reshape(xs, (C, batchs , -1))
        ys = np.reshape(ys, (C, batchs))
        epo = 0
        xCache = [(uvar.BaseVar(data=xs[i]), ys[i]) for i in range(C)]
        loss_hist = []
        accu_hist = []
        while epo < epochs:
            epo += 1
            loss = 0
            acc = 0
            for x, y in xCache:
                out = self.forward(x)
                l, dout = svm_loss(out.data, y)
                self.backward(uvar.BaseVar(grad=dout))
                self.optim(learning_rate)
                loss += l
                acc += np.sum(np.argmax(out.data, axis=1) == y)
            if epo % display_per_epoch == 0:
                print("epochs", epo, "loss", loss, "accuracy", acc/N)
                learning_rate *= 0.8
            loss_hist.append(loss)
            accu_hist.append(acc/N)
        return loss_hist, accu_hist

    def runModel(self, xs):
        """
        - xs: a numpy array, shape of (N, W, H)
        where N is the number of datas
        W is the width of image
        H is the height of image
        return the probolity of each class
        """
        N, _, _ = xs.shape
        x = np.reshape(xs, (N, -1))
        out = self.forward(uvar.BaseVar(data=x))
        return out.data


class AfineLayer():
    def __init__(self, W, O, alpfa=1, beta=0):
        """
        - W: input width
        - O: output size
        - alpfa, beta: in w initialization alpha*randn(M,O)+beta
        - self: 
            - w: (M,O) 
            - b: (O,)
        """
        w = uvar.RandnVar((W, O))
        b = uvar.RandnVar((O,))
        w.m, w.v, w.t = (None, None, 1)
        b.m, b.v, b.t = (None, None, 1)
        self.w, self.b = w, b
        self.out = uvar.BaseVar()

    def forward(self, x):
        """
        -x : a BaseVar
        """
        self.x = x
        # print("self fc", type(x))
        self.out.data = np.dot(x.data, self.w.data) + self.b.data
        return self.out

    def backward(self, dout):
        x, w, b = self.x, self.w, self.b
        dout = dout.grad
        # print(type(x), x)
        dx = np.dot(dout, w.data.T)
        dw = np.dot(x.data.T, dout)
        db = np.sum(dout, axis=0, keepdims=True)
        x.grad, w.grad, b.grad = dx, dw, np.reshape(db, b.data.shape)
        return x

    def optim(self, learning_rate):
        w, b = self.w, self.b
        # print("fc layer, optim learning_rate", learning_rate)
        w.data, w.m, w.v, w.t = adam(w.data, w.grad, learning_rate=learning_rate, m=w.m, v=w.v, t=w.t)
        b.data, b.m, b.v, b.t = adam(b.data, b.grad, learning_rate=learning_rate, m=b.m, v=b.v, t=b.t)


class ReLULayer():
    def __init__(self):
        self.out = uvar.BaseVar()

    def forward(self, x):
        """
        - x: a BaseVar
        return a BaseVar
        """
        # print("self ReLU", type(x))
        self.x = x
        self.out.data = np.where(x.data > 0, x.data, 0)
        return self.out

    def backward(self, dout):
        x = self.x
        dx = np.where(x.data > 0, dout.grad, 0)
        x.grad = dx
        return x
