import sys,os,argparse

import math
import numpy as np
import scipy.stats as sstat
import scipy.linalg as slin
import scipy.special as ssp
import sklearn.gaussian_process as GP

class naive_BO:
	def __init__(self, kernel, conf):
		self.init_kernel_ = kernel
		self.kernel_ = kernel
		self.floor_ = conf['floor']

		self.gp_ = None

	def fit(self, X_nd, Y_n, **kwargs):
		# Y_n[np.logical_not(np.isfinite(Y_n))] = self.floor_
		_Y_n = Y_n.copy()

		if (self.floor_ is None) or (self.floor_ is np.nan):
			fY_n = _Y_n[np.isfinite(_Y_n)]
			if fY_n.size > 0:
				_Y_n[np.isnan(_Y_n)] = np.min(fY_n)
			else:
				_Y_n[np.isnan(_Y_n)] = 0
		else:
			_Y_n[np.isnan(_Y_n)] = self.floor_

		return self._fit(X_nd, _Y_n)

	def _fit(self, X_nd, Y_n):
		gp = GP.GaussianProcessRegressor(kernel=self.init_kernel_, n_restarts_optimizer=4, normalize_y=False)

		gp.fit(X_nd, Y_n)

		self.gp_ = gp

		return gp

	def predict(self, X_nd):
		m_n, s_n = self.gp_.predict(X_nd, return_std=True)
		return m_n, s_n
		
	def acquisition(self, X_nd, return_prob=False, return_ms=False, **kwargs):
		if self.gp_ is None:
			return np.random.uniform(size=(X_nd.shape[0]))

		m_n, s_n = self.gp_.predict(X_nd, return_std=True)

		diff_best_n = (m_n - self.gp_.y_train_.max())
		z_n = diff_best_n / s_n

		cdf_n = sstat.norm.cdf(z_n)
		pdf_n = sstat.norm.pdf(z_n)

		EI_n = np.fmax(0, diff_best_n*cdf_n + s_n*pdf_n)

		ret = (EI_n, np.ones_like(EI_n) if return_prob else None, m_n if return_ms else None, s_n if return_ms else None)
		ret = tuple([elem for elem in ret if elem is not None])

		if len(ret)==1:
			return EI_n
		else:
			return ret

	def ucb(self, X_nd, return_prob=False, return_ms=False, scale=.1, **kwargs):
		if self.gp_ is None:
			return np.random.uniform(size=(X_nd.shape[0]))

		m_n, s_n = self.gp_.predict(X_nd, return_std=True)
		ucb_n = m_n + scale * s_n

		ret = (ucb_n, np.ones_like(ucb_n) if return_prob else None, m_n if return_ms else None, s_n if return_ms else None)
		ret = tuple([elem for elem in ret if elem is not None])

		if len(ret)==1:
			return ucb_n
		else:
			return ret


class variable_prior_BO:
	def __init__(self, kernel, conf):
		self.init_kernel_ = kernel
		self.kernel_ = kernel
		self.floor_ = conf['floor']
		self.alpha_ = conf['alpha']
		self.prior_ = 0

		self.gp_ = None

	def fit(self, X_nd, Y_n):
		_Y_n = Y_n.copy()

		if self.floor_ is None:
			fY_n = _Y_n[np.isfinite(_Y_n)]
			if fY_n.size > 0:
				self.prior_ = np.min(fY_n)
			else:
				self.prior_ = 0
		else:
			self.prior_ = self.floor_

		_Y_n[np.isnan(_Y_n)] = self.prior_

		self.prior_ = 0
		return self._fit(X_nd, _Y_n - self.prior_)

	def _fit(self, X_nd, Y_n):
		gp = GP.GaussianProcessRegressor(kernel=self.init_kernel_, n_restarts_optimizer=4, normalize_y=False, alpha=self.alpha_)

		gp.fit(X_nd, Y_n)

		self.gp_ = gp

		return gp

	def _trunc_acq(self, m_ni, s_ni, thres=0):
		# m_ni, s_ni, prior_i, u_i = self._pred_prior(X_nd, num_trials)

		# raw EI
		diff_best_ni = (m_ni - self.gp_.y_train_.max())
		z_ni = diff_best_ni / s_ni

		cdf_ni = sstat.norm.cdf(z_ni)
		pdf_ni = sstat.norm.pdf(z_ni)

		# raw_EI_ni = np.fmax(0, diff_best_ni*cdf_ni + s_ni*pdf_ni)

		# truncated
		trunc_diff = m_ni - thres
		trunc_z_ni = diff_best_ni / s_ni
		trunc_cdf_ni = sstat.norm.cdf(trunc_z_ni)
		trunc_pdf_ni = sstat.norm.pdf(trunc_z_ni)

		trunc_EI_ni = np.fmax(0, diff_best_ni*cdf_ni + s_ni*pdf_ni - trunc_diff*trunc_cdf_ni - s_ni*trunc_pdf_ni)

		return trunc_EI_ni

	def predict(self, X_nd):
		m_n, s_n = self.gp_.predict(X_nd, return_std=True)
		m_n += self.prior_

		return m_n, s_n

	def acquisition(self, X_nd, return_prob=False, return_ms=False, **kwargs):
		if self.gp_ is None:
			return np.random.uniform(size=(X_nd.shape[0]))

		m_n, s_n = self.gp_.predict(X_nd, return_std=True)
		m_n += self.prior_

		diff_best_n = (m_n - (self.gp_.y_train_.max() + self.prior_))
		z_n = diff_best_n / s_n

		cdf_n = sstat.norm.cdf(z_n)
		pdf_n = sstat.norm.pdf(z_n)

		EI_n = np.fmax(0, diff_best_n*cdf_n + s_n*pdf_n)

		ret = (EI_n, 
			np.ones_like(EI_n) if return_prob else None, 
			m_n if return_ms else None, s_n if return_ms else None, 
			self.prior_ if 'return_prob' in kwargs and kwargs['return_prob'] else None)
		ret = tuple([elem for elem in ret if elem is not None])

		if len(ret)==1:
			return EI_n
		else:
			return ret

		# return (EI_n, np.ones(EI_n.size)) if return_prob else EI_n

	def trunc_acquisition(self, X_nd, return_prob=False, return_ms=False, num_trials=1):
		if self.gp_ is None:
			return np.random.uniform(size=(X_nd.shape[0]))

		m_n, s_n = self.gp_.predict(X_nd, return_std=True)
		m_n += self.prior_
		EI_n = self._trunc_acq(m_n, s_n)

		ret = (EI_n, 
			np.ones_like(EI_n) if return_prob else None, 
			m_n if return_ms else None, s_n if return_ms else None)
		ret = tuple([elem for elem in ret if elem is not None])

		if len(ret)==1:
			return EI_n
		else:
			return ret



class adaptive_leveling_BO(variable_prior_BO):
	def __init__(self, kernel, conf):
		super(adaptive_leveling_BO, self).__init__(kernel, conf)
		self.last_prior_ = None


	def fit(self, X_nd, Y_n, t_num=None):
		if t_num is None or type(t_num) is not tuple:
			val_n = np.isfinite(Y_n)
			num_val = np.sum(val_n)
			num_nan = np.sum(1 - val_n)

			self.beta_ = (num_val if num_val > 0 else .1, num_nan if num_nan > 0 else .1)
		else:
			self.beta_ = t_num[:2]

		# Y_n[np.logical_not(np.isfinite(Y_n))] = self.floor_
		_Y_n = Y_n.copy()

		# floor padding
		if self.floor_ is None:
			fY_n = _Y_n[np.isfinite(_Y_n)]
			if fY_n.size > 0:
				self.floor_ = np.min(fY_n)
			else:
				self.floor_ = 0

		_Y_n[np.isnan(_Y_n)] = self.floor_

		self.prior_ = 0

		self.gp_ = GP.GaussianProcessRegressor(kernel=self.init_kernel_, n_restarts_optimizer=1, normalize_y=False, alpha=self.alpha_)
		self.gp_.fit(X_nd, _Y_n)

		return self.gp_

	def predict(self, X_nd, num_trials=1):
		m_ni = np.zeros((X_nd.shape[0], num_trials))
		s_ni = np.zeros_like(m_ni)

		# u_i = np.random.beta(self.beta_[0], self.beta_[1], size=num_trials)
		u_i = np.random.uniform(size=num_trials)
		ceil = np.nanmax(self.gp_.y_train_)
		floor = np.nanmin(self.gp_.y_train_)

		for i in range(num_trials):
			# prior level
			prior = floor + u_i[i] * (ceil - floor)

			# GP fitting
			gp = GP.GaussianProcessRegressor(kernel=self.init_kernel_, n_restarts_optimizer=5, normalize_y=False, alpha=self.alpha_)
			gp.fit(self.gp_.X_train_, self.gp_.y_train_ - prior)

			m, s = gp.predict(X_nd, return_std=True)

			m_ni[:,i] = m + prior
			s_ni[:,i] = s

		return np.squeeze(m_ni), np.squeeze(s_ni)

	def _pred_prior(self, X_nd, num_trials=1):
		m_ni = np.zeros((X_nd.shape[0], num_trials))
		s_ni = np.zeros_like(m_ni)

		u_i = np.random.uniform(size=num_trials)
		ceil = np.nanmax(self.gp_.y_train_)
		prior_i = self.floor_ + u_i * (ceil - self.floor_)

		for i in range(num_trials):
			# prior level
			prior = prior_i[i]

			# GP fitting
			gp = GP.GaussianProcessRegressor(kernel=self.init_kernel_, n_restarts_optimizer=5, normalize_y=False, alpha=self.alpha_)
			gp.fit(self.gp_.X_train_, self.gp_.y_train_ - prior)

			m, s = gp.predict(X_nd, return_std=True)

			m_ni[:,i] = m + prior
			s_ni[:,i] = s

		return np.squeeze(m_ni), np.squeeze(s_ni), prior_i, u_i

	def _acq_prior(self, X_nd, num_trials=1):
		if self.gp_ is None:
			return np.random.uniform(size=(X_nd.shape[0]))

		m_ni, s_ni, prior_i, u_i = self._pred_prior(X_nd, num_trials)

		diff_best_ni = (m_ni - self.gp_.y_train_.max() - prior_i[None,:])
		z_ni = diff_best_ni / s_ni

		cdf_ni = sstat.norm.cdf(z_ni)
		pdf_ni = sstat.norm.pdf(z_ni)

		EI_ni = np.fmax(0, diff_best_ni*cdf_ni + s_ni*pdf_ni)

		return EI_ni, prior_i, u_i


	def acquisition(self, X_nd, return_prob=False, return_ms=False, return_prior=False, num_trials=1):
		if self.gp_ is None:
			return np.random.uniform(size=(X_nd.shape[0]))

		m_ni, s_ni, prior_i, u_i = self._pred_prior(X_nd, num_trials)
		self.last_prior_ = np.mean(prior_i)

		diff_best_ni = (m_ni - self.gp_.y_train_.max())# - prior_i[None,:])
		z_ni = diff_best_ni / s_ni

		cdf_ni = sstat.norm.cdf(z_ni)
		pdf_ni = sstat.norm.pdf(z_ni)

		EI_ni = np.fmax(0, diff_best_ni*cdf_ni + s_ni*pdf_ni)

		ret = (EI_ni, 
			np.ones_like(EI_ni) if return_prob else None, 
			m_ni if return_ms else None, s_ni if return_ms else None, 
			prior_i if return_prior else None)
		ret = tuple([elem for elem in ret if elem is not None])

		if len(ret)==1:
			return EI_ni
		else:
			return ret

	def trunc_acquisition(self, X_nd, return_prob=False, return_ms=False, return_prior=False, num_trials=1):
		if self.gp_ is None:
			return np.random.uniform(size=(X_nd.shape[0]))

		m_ni, s_ni, prior_i, u_i = self._pred_prior(X_nd, num_trials)
		EI_ni = self._trunc_acq(m_ni, s_ni)
		self.last_prior_ = np.mean(prior_i)

		ret = (EI_ni, 
			np.ones_like(EI_ni) if return_prob else None, 
			m_ni if return_ms else None, s_ni if return_ms else None, 
			prior_i if return_prior else None)
		ret = tuple([elem for elem in ret if elem is not None])

		if len(ret)==1:
			return EI_ni
		else:
			return ret

	def ucb(self, X_nd, return_prob=False, return_ms=False, num_trials=1, scale=.1):
		if self.gp_ is None:
			return np.random.uniform(size=(X_nd.shape[0]))

		m_ni, s_ni = self.predict(X_nd, num_trials)
		ucb_ni = m_ni + s_ni * scale

		ret = (ucb_ni, np.ones_like(ucb_ni) if return_prob else None, m_ni if return_ms else None, s_ni if return_ms else None)
		ret = tuple([elem for elem in ret if elem is not None])

		if len(ret)==1:
			return ucb_ni
		else:
			return ret


class EB_mean_prior_BO(variable_prior_BO):
	def __init__(self, kernel, conf):
		super(EB_mean_prior_BO, self).__init__(kernel, conf)
		self.last_prior_ = None
		self.prior_ = 0

	def fit(self, X_nd, Y_n, **kwargs):
		_Y_n = Y_n.copy()

		# floor padding
		if self.floor_ is None:
			fY_n = _Y_n[np.isfinite(_Y_n)]
			if fY_n.size > 0:
				self.floor_ = np.min(fY_n)
			else:
				self.floor_ = 0

		_Y_n[np.isnan(_Y_n)] = self.floor_


		if self.gp_ is None:
			# initial fitting
			gp = GP.GaussianProcessRegressor(kernel=self.init_kernel_, n_restarts_optimizer=1, normalize_y=False, alpha=self.alpha_)
			gp.fit(X_nd, _Y_n - self.prior_)

			K_nn = gp.kernel_(X_nd)
			invK_ones = np.linalg.solve(K_nn, np.ones(K_nn.shape[0]))
			prior = np.sum(invK_ones * _Y_n) / np.sum(invK_ones)

			gp.fit(X_nd, _Y_n - prior)
			self.gp_ = gp
			self.prior_ = prior

		else:
			# accessible to previous prior
			K_nn = self.gp_.kernel_(X_nd)
			invK_ones = np.linalg.solve(K_nn, np.ones(K_nn.shape[0]))
			prior = np.sum(invK_ones * _Y_n) / np.sum(invK_ones)

			self.gp_.fit(X_nd, _Y_n - prior)
			self.prior_ = prior


		return self.gp_

	def predict(self, X_nd):
		m_n, s_n = self.gp_.predict(X_nd, return_std=True)

		return m_n + self.prior_, s_n

	def acquisition(self, X_nd, return_prob=False, return_ms=False, return_prior=False, **kwargs):
		if self.gp_ is None:
			return np.random.uniform(size=(X_nd.shape[0]))

		m_n, s_n = self.predict(X_nd,)

		diff_best_n = (m_n - (self.gp_.y_train_.max() + self.prior_))
		z_n = diff_best_n / s_n

		cdf_n = sstat.norm.cdf(z_n)
		pdf_n = sstat.norm.pdf(z_n)

		EI_n = np.fmax(0, diff_best_n*cdf_n + s_n*pdf_n)

		ret = (EI_n, 
			np.ones_like(EI_n) if return_prob else None, 
			m_n if return_ms else None, s_n if return_ms else None, 
			self.prior_ if return_prior else None)
		ret = tuple([elem for elem in ret if elem is not None])

		if len(ret)==1:
			return EI_n
		else:
			return ret

class arithmetic_mean_prior_BO(variable_prior_BO):
	def __init__(self, kernel, conf):
		super(arithmetic_mean_prior_BO, self).__init__(kernel, conf)
		self.last_prior_ = None
		self.prior_ = 0

	def fit(self, X_nd, Y_n, **kwargs):
		_Y_n = Y_n.copy()

		# floor padding
		if self.floor_ is None:
			fY_n = _Y_n[np.isfinite(_Y_n)]
			if fY_n.size > 0:
				self.floor_ = np.min(fY_n)
			else:
				self.floor_ = 0

		_Y_n[np.isnan(_Y_n)] = self.floor_


		if self.gp_ is None:
			# initial fitting
			gp = GP.GaussianProcessRegressor(kernel=self.init_kernel_, n_restarts_optimizer=1, normalize_y=False, alpha=self.alpha_)
			prior = np.mean(_Y_n)

			gp.fit(X_nd, _Y_n - prior)
			self.gp_ = gp
			self.prior_ = prior

		else:
			# accessible to previous prior
			prior = np.mean(_Y_n)

			self.gp_.fit(X_nd, _Y_n - prior)
			self.prior_ = prior


		return self.gp_

	def predict(self, X_nd):
		m_n, s_n = self.gp_.predict(X_nd, return_std=True)

		return m_n + self.prior_, s_n

	def acquisition(self, X_nd, return_prob=False, return_ms=False, return_prior=False, **kwargs):
		if self.gp_ is None:
			return np.random.uniform(size=(X_nd.shape[0]))

		m_n, s_n = self.predict(X_nd,)

		diff_best_n = (m_n - (self.gp_.y_train_.max() + self.prior_))
		z_n = diff_best_n / s_n

		cdf_n = sstat.norm.cdf(z_n)
		pdf_n = sstat.norm.pdf(z_n)

		EI_n = np.fmax(0, diff_best_n*cdf_n + s_n*pdf_n)

		ret = (EI_n, 
			np.ones_like(EI_n) if return_prob else None, 
			m_n if return_ms else None, s_n if return_ms else None, 
			self.prior_ if return_prior else None)
		ret = tuple([elem for elem in ret if elem is not None])

		if len(ret)==1:
			return EI_n
		else:
			return ret

class EB_uniform_prior_BO(variable_prior_BO):
	def __init__(self, kernel, conf):
		super(EB_uniform_prior_BO, self).__init__(kernel, conf)
		self.last_prior_ = None
		self.prior_ = 0
		self.prior_width_ = None

	def fit(self, X_nd, Y_n, **kwargs):
		_Y_n = Y_n.copy()

		# floor padding
		if self.floor_ is None:
			fY_n = _Y_n[np.isfinite(_Y_n)]
			if fY_n.size > 0:
				self.floor_ = np.min(fY_n)
			else:
				self.floor_ = 0

		_Y_n[np.isnan(_Y_n)] = self.floor_

		if self.gp_ is None:
			# initial fitting
			gp = GP.GaussianProcessRegressor(kernel=self.init_kernel_, n_restarts_optimizer=1, normalize_y=False, alpha=self.alpha_)
			gp.fit(X_nd, _Y_n - self.prior_)

			K_nn = gp.kernel_(X_nd)
			invK_ones = np.linalg.solve(K_nn, np.ones(K_nn.shape[0]))
			prior_width = np.sum(invK_ones * _Y_n) / np.sum(invK_ones)

			self.prior_width_ = prior_width

		else:
			K_nn = self.gp_.kernel_(X_nd)
			invK_ones = np.linalg.solve(K_nn, np.ones(K_nn.shape[0]))
			prior_width = np.sum(invK_ones * _Y_n) / np.sum(invK_ones)

			self.prior_width_ = prior_width

		gp = GP.GaussianProcessRegressor(kernel=self.init_kernel_, n_restarts_optimizer=1, normalize_y=False, alpha=self.alpha_)
		prior = np.random.uniform(low=0, high=np.fabs(self.prior_width_ * 2)) * np.sign(self.prior_width_)

		self.prior_ = prior
		gp.fit(X_nd, _Y_n - self.prior_)
		self.gp_ = gp

	def predict(self, X_nd, num_trials=1):
		m_ni = np.zeros((X_nd.shape[0], num_trials))
		s_ni = np.zeros_like(m_ni)

		u_i = np.random.uniform(size=num_trials)
		ceil = self.prior_width_ * 2
		floor = 0

		for i in range(num_trials):
			# prior level
			prior = floor + u_i[i] * (ceil - floor)

			# GP fitting
			gp = GP.GaussianProcessRegressor(kernel=self.init_kernel_, n_restarts_optimizer=5, normalize_y=False, alpha=self.alpha_)
			gp.fit(self.gp_.X_train_, self.gp_.y_train_ + self.prior_ - prior)

			m, s = gp.predict(X_nd, return_std=True)

			m_ni[:,i] = m + prior
			s_ni[:,i] = s

		return np.squeeze(m_ni), np.squeeze(s_ni)

	def _pred_prior(self, X_nd, num_trials=1):
		m_ni = np.zeros((X_nd.shape[0], num_trials))
		s_ni = np.zeros_like(m_ni)

		u_i = np.random.uniform(size=num_trials)
		ceil = self.prior_width_ * 2
		prior_i =  u_i * ceil

		for i in range(num_trials):
			# prior level
			prior = prior_i[i]

			# GP fitting
			gp = GP.GaussianProcessRegressor(kernel=self.init_kernel_, n_restarts_optimizer=5, normalize_y=False, alpha=self.alpha_)
			gp.fit(self.gp_.X_train_, self.gp_.y_train_ + self.prior_ - prior)

			# if i < 1:
			# 	ms, s_n = gp.predict(X_nd, return_std=True)
			# else:
			# 	ms = gp.predict(X_nd)
			m, s = gp.predict(X_nd, return_std=True)

			m_ni[:,i] = m + prior
			s_ni[:,i] = s

		return np.squeeze(m_ni), np.squeeze(s_ni), prior_i, u_i

	def acquisition(self, X_nd, return_prob=False, return_ms=False, return_prior=False, num_trials=1):
		if self.gp_ is None:
			return np.random.uniform(size=(X_nd.shape[0]))

		# m_ni, s_ni = self.predict(X_nd, num_trials)
		m_ni, s_ni, prior_i, u_i = self._pred_prior(X_nd, num_trials)

		diff_best_ni = (m_ni - self.gp_.y_train_.max() - self.prior_)
		z_ni = diff_best_ni / s_ni

		cdf_ni = sstat.norm.cdf(z_ni)
		pdf_ni = sstat.norm.pdf(z_ni)

		EI_ni = np.fmax(0, diff_best_ni*cdf_ni + s_ni*pdf_ni)

		ret = (EI_ni, 
			np.ones_like(EI_ni) if return_prob else None, 
			m_ni if return_ms else None, s_ni if return_ms else None, 
			prior_i if return_prior else None)
		ret = tuple([elem for elem in ret if elem is not None])

		if len(ret)==1:
			return EI_ni
		else:
			return ret

	def _acq_prior(self, X_nd, num_trials=1):
		if self.gp_ is None:
			return np.random.uniform(size=(X_nd.shape[0]))

		m_ni, s_ni, prior_i, u_i = self._pred_prior(X_nd, num_trials)

		diff_best_ni = (m_ni - self.gp_.y_train_.max() - prior_i[None,:])
		z_ni = diff_best_ni / s_ni

		cdf_ni = sstat.norm.cdf(z_ni)
		pdf_ni = sstat.norm.pdf(z_ni)

		EI_ni = np.fmax(0, diff_best_ni*cdf_ni + s_ni*pdf_ni)

		return EI_ni, prior_i, u_i
