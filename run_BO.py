import sys,os,argparse,tqdm

import math
import numpy as np
import sklearn.gaussian_process as GP
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import BO_core, obj_func, utils, visualize


if __name__=='__main__':
	parser = argparse.ArgumentParser(description='BO')
	parser.add_argument('--obsvar', type=float, default=0.001, help='variance of observation noise')
	parser.add_argument('--num', type=int, default=100)
	parser.add_argument('--init', type=int, default=5)
	parser.add_argument('--repeat', type=int, default=1)
	# search space
	parser.add_argument('--finegrid', type=int, help='number of grid cells in each dimension. Lower this value if memory runs out.')
	parser.add_argument('--dim', type=int, default=2)

	parser.add_argument('--obj', choices=['ac','rb', 'ackley', 'rosenbrock'], default='ac')
	parser.add_argument('--mode', choices=['base', 'da', 'al', 'eb', 'ebu', 'DA', 'AL', 'EB', 'EBu'], default='al')

	parser.add_argument('--nacq', type=int, default=1)
	parser.add_argument('--bias', type=float, default=0.5)

	parser.add_argument('--seed', type=int, default=3210)
	parser.add_argument('--out', help='filename to save results. Stored in .npz format.')

	args = parser.parse_args()

	if args.obj.lower() in ['ac', 'ackley']:
		func = obj_func.ackley(args.dim, args.bias)
	else:
		func = obj_func.rosenbrock(args.dim, args.bias)

	D = args.dim

	np.random.seed(args.seed)

	xbb = np.zeros((2,D)) # search space: [-0.5, 2]^D box
	xbb[0,:] = -0.5
	xbb[1,:] = 2
	ybb = np.asarray([[0],[1]]) # shape=(2,1)

	# BO configurations
	kernel = GP.kernels.Product(GP.kernels.ConstantKernel(constant_value=1., constant_value_bounds=(1e-1,1e1)), GP.kernels.Matern(length_scale=1, length_scale_bounds=(1e-2, .1),nu=2.5))
	kernel = GP.kernels.Sum(kernel, GP.kernels.WhiteKernel(1e-4, noise_level_bounds=(1e-5,1e-1)))
	if args.mode.lower() == 'al':
		BOclass = BO_core.adaptive_leveling_BO
	elif args.mode.lower() == 'da':
		BOclass = BO_core.arithmetic_mean_prior_BO
	elif args.mode.lower() == 'eb':
		BOclass = BO_core.EB_mean_prior_BO
	elif args.mode.lower() == 'ebu':
		BOclass = BO_core.EB_uniform_prior_BO
	else:
		BOclass = BO_core.naive_BO
	if D < 4: # acquisition function evaluates num_acq_grid**D points
		num_acq_grid = 50
	else:
		num_acq_grid = 10
	if args.finegrid is not None:
		num_acq_grid = int(args.finegrid)

	# results
	y_rn = np.zeros((args.repeat, args.num))
	x_rnd = np.zeros((args.repeat, args.num, D))



	# entering loop here
	for r in tqdm.tqdm(range(args.repeat), desc='Runs'):
		# initial observations
		raw_config_nd = utils.initial_x(args.init, args.seed, xbb) # input x
		rawy_n = func(raw_config_nd) # output y
		noise_level = math.sqrt(args.obsvar)
		eps_n = np.random.normal(size=rawy_n.size) * noise_level

		opt_progress_bar = tqdm.tqdm(total=args.num, leave=False, desc='observations')
		opt_progress_bar.update(raw_config_nd.shape[0])
		# incremental BO search
		while raw_config_nd.shape[0] < args.num:
			# normalized values
			X_nd = utils.all_r2u(raw_config_nd, xbb)
			Y_n = utils.raw_to_uni(rawy_n + eps_n, ybb)

			# BO setup
			simplefilter('ignore', category=ConvergenceWarning)
			bo = BOclass(kernel, {'floor':None, 'alpha':0})

			# fitting
			bo.fit(X_nd, Y_n, t_num=None)

			# one-shot acquisition
			grid_X_nd = utils.fixed_grid_uniform(np.asarray((num_acq_grid,)*D), [])
			acq_ni, m_ni, s_ni = bo.acquisition(grid_X_nd, return_ms=True, num_trials=args.nacq) # expected improvement (EI)

			NUM = int(np.sqrt(acq_ni.shape[0]))

			if acq_ni.ndim<2:
				idx_n = np.argsort(acq_ni)[::-1]

			else:
				idx_n = np.argsort(np.mean(acq_ni, axis=1))[::-1]

			# observe function value
			raw_config_nd = np.r_[raw_config_nd, np.asarray(utils.uni_to_raw(grid_X_nd[idx_n[0],:], xbb))[None,:]]
			rawy_n = np.r_[rawy_n, func(raw_config_nd[-1,:])]
			eps_n = np.r_[eps_n, np.random.normal(size=1) * noise_level]

			opt_progress_bar.update(1)

		y_rn[r,:] = rawy_n
		x_rnd[r,:,:] = raw_config_nd


	if args.out is not None and len(args.out) > 0:
		np.savez(args.out, x=x_rnd, y=y_rn)

	sns.set_style('whitegrid')
	visualize.fill_plot(visualize.get_cummax(y_rn))
	plt.show()