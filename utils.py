
import numpy as np

import lhsmdu


###################################
# acquisition utility
###################################
def grid_uniform(space_d):
	grid_d = np.meshgrid(*space_d, indexing='ij')

	for d in range(len(grid_d)):
		# calculate interval
		diff = np.diff(sorted(list(set(grid_d[d].ravel()))))
		if diff.size == 0:
			continue
		interval = np.mean(diff)

		if np.isfinite(interval):
			grid_d[d] += np.random.uniform(high=interval, size=grid_d[d].shape)

	return np.c_[[g.ravel() for g in grid_d]].T

def num_grid_uniform(num_d, low=0, high=1):
	space_d = [np.linspace(low,high,num+1)[:num] for num in num_d]

	return grid_uniform(space_d)

def fixed_grid_uniform(num_d, l_fix, low=0, high=1):
	space_d = [np.linspace(low,high,num+1)[:num] for num in num_d]
	for d,val in l_fix:
		space_d[d] = np.asarray([val])

	return grid_uniform(space_d)


##################################
# Latin hypercube sampling
##################################
def lhs_uniform(x_d, len_d, l_fix=[], num=64):
	uni01_nd = np.asarray(lhsmdu.sample(x_d.size, num)).T
	grid_X_nd = np.clip(x_d[np.newaxis,:] + len_d[np.newaxis,:] * (uni01_nd - 0.5), a_min=0, a_max=1)

	for d,val in l_fix:
		grid_X_nd[:,d] = val

	return grid_X_nd


####################################
# [0,1] -- raw config value convertor
# x: [0,1]
# raw: values used in actual experiments
####################################
def uni_to_raw(x_d, bb):
	return bb[0,:] + x_d * (bb[1,:] - bb[0,:])

def raw_to_uni(raw_d, bb):
	return (raw_d - bb[0,:]) / (bb[1,:] - bb[0,:])

def all_u2r(x_nd, bb):
	return np.asarray([uni_to_raw(x_nd[n,:], bb) for n in range(x_nd.shape[0])])

def all_r2u(raw_config_nd, bb):
	return np.asarray([raw_to_uni(raw_config_nd[n,:], bb) for n in range(raw_config_nd.shape[0])])

#####################################
# misc
#####################################
def initial_x(num, seed, xbb):
	return all_u2r(np.asarray(lhsmdu.sample(xbb.shape[1], num, randomSeed=seed)).T, xbb)

if __name__=='__main__':
	print('utils.py for internal functions')