import numpy as np


def affine_forward(x, w, b):
	"""
	Computes the forward pass for an affine (fully-connected) layer.

	Inputs:
	- x: A numpy array containing input data, of shape (N, D)
	- w: A numpy array of weights, of shape (D, M)
	- b: A numpy array of biases, of shape (M,)

	Returns a tuple of:
	- out: output, of shape (N, M)
	- cache: (x, w)
	"""
	out = np.dot(x, w) + b
	return out, x, w

def affine_backward(dout, x, w):
	"""
	Computes the backward pass for an affine layer.

	Inputs:
	- dout: Upstream derivative, of shape (N, M)
	- x: Input data, of shape (N, D)
	- w: Weights, of shape (D, M)

	Returns a tuple of:
	- dx: Gradient with respect to x, of shape (N, D)
	- dw: Gradient with respect to w, of shape (D, M)
	- db: Gradient with respect to b, of shape (M,)
	"""
	dx = np.dot(dout, w.T)
	dw = np.dot(x.T, dout)
	db = np.sum(dout, axis=0, keepdims = True)
	return dx, dw, db

def relu_forward(x):
	"""
	Computes the forward pass for a layer of rectified linear units (ReLUs).

	Input:
	- x: Inputs, of any shape

	Returns a tuple of:
	- out: Output, of the same shape as x
	- cache: x
	"""	
	out = np.where(x > 0, x, 0)
	return out, x


def relu_backward(dout, cache):
	"""
	Computes the backward pass for a layer of rectified linear units (ReLUs).

	Input:
	- dout: Upstream derivatives, of any shape
	- cache: Input x, of same shape as dout

	Returns:
	- dx: Gradient with respect to x
	"""
	return np.where(cache > 0, dout, 0)


def svm_loss(x, y):
	"""
	Computes the loss and gradient using for multiclass SVM classification.

	Inputs:
	- x: Input data, of shape (N, C) where x[i, j] is the score for the jth
	  class for the ith input.
	- y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
	  0 <= y[i] < C

	Returns a tuple of:
	- loss: Scalar giving the loss
	- dx: Gradient of the loss with respect to x
	"""
	N = x.shape[0]
	correct_class_scores = x[np.arange(N), y]
	margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
	margins[np.arange(N), y] = 0
	loss = np.sum(margins) / N
	num_pos = np.sum(margins > 0, axis=1)
	dx = np.zeros_like(x)
	dx[margins > 0] = 1
	dx[np.arange(N), y] -= num_pos
	dx /= N
	return loss, dx


def softmax_loss(x, y):
	"""
	Computes the loss and gradient for softmax classification.

	Inputs:
	- x: Input data, of shape (N, C) where x[i, j] is the score for the jth
	  class for the ith input.
	- y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
	  0 <= y[i] < C

	Returns a tuple of:
	- loss: Scalar giving the loss
	- dx: Gradient of the loss with respect to x
	"""
	shifted_logits = x - np.max(x, axis=1, keepdims=True)
	Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
	log_probs = shifted_logits - np.log(Z)
	probs = np.exp(log_probs)
	N = x.shape[0]
	loss = -1 * np.sum(log_probs[np.arange(N), y]) / N
	dx = probs.copy()
	dx[np.arange(N), y] -= 1
	dx /= N
	return loss, dx
