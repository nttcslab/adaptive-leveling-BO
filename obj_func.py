import sys,os,argparse

import numpy as np
import scipy as sp
import scipy.special as spf

import matplotlib.pyplot as plt
import seaborn as sns


# def rosenbrock_raw(x_nd, scale=100):
# 	if x_nd.ndim<2:
# 		x_nd = x_nd[None,:]

# 	y_n = np.sum((x_nd - 1)**2, axis=1) + scale * np.sum((x_nd[:,1:] - x_nd[:,:-1]**2)**2, axis=1)

# 	return -y_n


def rosenbrock_scale(x_nd, scale=10):
	if x_nd.ndim<2:
		x_nd = x_nd[None,:]

	const = (1/3.755) * 1e-8
	bias = 3.825 / 3.755

	y_n = const * np.sum((x_nd - 1)**2, axis=1) + scale * np.sum((x_nd[:,1:] - x_nd[:,:-1]**2)**2, axis=1)
	return bias - y_n


def ackley_scale(x_nd, a=2, b=0.2, c=4):
	if x_nd.ndim<2:
		x_nd = x_nd[None,:]

	y_n = a*np.exp(-b * np.sqrt(np.mean(x_nd**2, axis=1))) + np.exp(np.mean(np.cos(c*x_nd), axis=1)) - a - 2.718281828459

	return y_n

class ackley:
	def __init__(self, dim, bias=0.5):
		self.bias_ = bias
		if dim==2:
			self.level_ = 0.0003304
			self.scale_ = 1. / 2.678629
		elif dim==4:
			self.level_ = 0.0141873
			self.scale_ = 1. / 2.678055
		elif dim==6:
			self.level_ = 0.0141873
			self.scale_ = 1. / 2.678055
		else:
			self.level_ = 0
			self.scale_ = 1. / 2.68

	def __call__(self, x):
		return self.bias_ + (ackley_scale(x) + self.level_) * self.scale_

class rosenbrock:
	def __init__(self, dim, bias=0.5):
		self.bias_ = bias
		if dim==2:
			self.level_ = -1.01865
			self.scale_ = 1./ 101.0945
		elif dim==4:
			self.level_ = -1.01820
			self.scale_ = 1./ 204.98594
		elif dim==6:
			self.level_ = -0.95844
			self.scale_ = 1./ 290.2418
		else:
			self.level_ = -1
			self.scale_ = 1./ (50*dim)


	def __call__(self, x):
		return self.bias_ + (rosenbrock_scale(x) + self.level_) * self.scale_

if __name__=='__main__':
	print('objective functions')