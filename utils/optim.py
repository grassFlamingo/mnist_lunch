import numpy as np


def sgd(w, dw, learning_rate=1e-2):
	"""
	Performs vanilla stochastic gradient descent.

	config format:
	- learning_rate: Scalar learning rate.
	"""
	w -= learning_rate * np.reshape(dw, w.shape)
	return w


def sgd_momentum(w, dw, velocity=None, learning_rate=1e-2, momentum=0.9):
	"""
	Performs stochastic gradient descent with momentum.

	config format:
	- learning_rate: Scalar learning rate.
	- momentum: Scalar between 0 and 1 giving the momentum value.
	  Setting momentum = 0 reduces to sgd.
	- velocity: A numpy array of the same shape as w and dw used to store a
	  moving average of the gradients.
	"""
	if velocity is None:
		velocity = np.zeros_like(w)
	velocity = momentum * velocity - learning_rate * dw
	next_w = w + velocity
	return next_w, velocity


def rmsprop(x,
			dx,
			cache=None,
			learning_rate=1e-2,
			decay_rate=0.99,
			epsilon=1e-8):
	"""
	Uses the RMSProp update rule, which uses a moving average of squared
	gradient values to set adaptive per-parameter learning rates.

	config format:
	- learning_rate: Scalar learning rate.
	- decay_rate: Scalar between 0 and 1 giving the decay rate for the squared
	  gradient cache.
	- epsilon: Small scalar used for smoothing to avoid dividing by zero.
	- cache: Moving average of second moments of gradients.
	"""
	if cache is None:
		cache = np.zeros_like(x)
	cache = decay_rate * cache + (1 - decay_rate) * (dx * dx)
	x -= learning_rate * dx / (np.sqrt(cache) + epsilon)
	return x, cache


def adam(x, dx, learning_rate=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8, m=None, v=None, t=1):
	"""
	Uses the Adam update rule, which incorporates moving averages of both the
	gradient and its square and a bias correction term.

	config format:
	- learning_rate: Scalar learning rate.
	- beta1: Decay rate for moving average of first moment of gradient.
	- beta2: Decay rate for moving average of second moment of gradient.
	- epsilon: Small scalar used for smoothing to avoid dividing by zero.
	- m: Moving average of gradient.
	- v: Moving average of squared gradient.
	- t: Iteration number.
	"""
	if m is None:
		m = np.zeros_like(x)
	if v is None:
		v = np.zeros_like(x)

	t = t + 1

	# Momentum
	m = beta1 * m + (1 - beta1) * dx

	# AdaGrad
	v = beta2 * v + (1 - beta2) * dx * dx

	mt = m / (1 - beta1**t)  # for batch normalization
	vt = v / (1 - beta2**t)  # for batch normalization

	# AdaGrid/RMSProp
	x -= learning_rate * mt / (np.sqrt(vt) + epsilon)
	return x, m, v, t
