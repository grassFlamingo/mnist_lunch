import numpy as np
import numpy.random

class BaseVar:
    data = None
    grad = None
    def __init__(self, data=None, grad=None):
        self.data = data
        self.grad = grad
    def __str__(self):
        return "BaseVar "

class ZeroVar(BaseVar):
   
    def __init__(self, shape):
        """
        - shape: tupe the shape of data type
        """
        self.data = np.zeros(shape)

class RandnVar(BaseVar):

    def __init__(self, shape, alpha=1, beta=0):
        """
        - shape: tupe the shape of data type
        """
        self.data = alpha * np.random.randn(*shape) + beta
