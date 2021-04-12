import numpy as np
from scipy.optimize import curve_fit


class SNLine:
	def __init__(self,
			  a_amp=1, a_fwhm=1, a_mean=0, 
			  b_amp=1, b_fwhm=1, b_mean=0, 
			  c_amp=1, c_fwhm=1, c_mean=0, add_red_absorption=False,
			  d_amp=0, d_fwhm=1, d_mean=0, add_another_gaussian=False,
			  wl=[], vel=[], fl=[], var=[], std=[], colour='black', date=""
			  ):
		#Date of observation
		self.date = date
		#Colour of contour
		self.colour = colour
		#Normalised flux values
		self.fl = fl
		self.fl_fit = np.zeros(len(fl))
		#Calculate proper wavelength from redshift
		self.wl = wl
		#Convert to velocity space
		self.vel = vel
		#Errors
		self.var = var
		self.std = std
		#Figure out continuum background to offset the plot by
		self.continuum_offset = self.continuumOffset()

	def continuumOffset(self):
		'''
		Calculates a constant continuum offset
		'''
		first_fl = np.nanmean(self.fl[:10])
		last_fl = np.nanmean(self.fl[-10:])

		return np.nanmean([first_fl, last_fl]) 

	#Trim wavelengths to within specified range
	def TrimWL(self,min_wl=0, max_wl=10000):
		i_min = np.searchsorted(self.wl, min_wl, side='left')
		i_max = np.searchsorted(self.wl, max_wl, side='left')
		self.wl = self.wl[i_min:i_max]
		self.fl = self.fl[i_min:i_max]
		self.fl_fit = self.fl[i_min:i_max]
		self.vel = self.vel[i_min:i_max]
		self.std = self.std[i_min:i_max]
		self.var = self.var[i_min:i_max]

	#Trim velocities to within specified range
	def TrimVel(self,min_vel=-1000, max_vel=1000):
		i_min = np.searchsorted(self.vel, min_vel, side='left')
		i_max = np.searchsorted(self.vel, max_vel, side='left')
		self.wl = self.wl[i_min:i_max]
		self.fl = self.fl[i_min:i_max]
		self.fl_fit = self.fl_fit[i_min:i_max]
		self.vel = self.vel[i_min:i_max]
		self.std = self.std[i_min:i_max]
		self.var = self.var[i_min:i_max]

	def Scale(self, factor=1):
		self.fl = np.multiply(self.fl, factor)
		self.fl_fit = np.multiply(self.fl_fit, factor)
		self.std = np.multiply(self.std, factor)
		self.var = np.multiply(self.var, np.power(factor,2))


	def fitCurve(self, *argv, **kwargs):
		'''
		Calculates error in val_2 if same percentile error as val_1

		Parameters
		----------
		**kwargs:
			gaussians 	:	(dict) {'gaussian_num': {'amp': X, 'fwhm': Y, 'mean':Z}}
							Dictionary holding important vales for each gaussian to fit
			amp_percent_range	:	(float)
			fwhm_percent_range		Percentage to alter initial values by for curve_fit bounds
			continuum_offset_percent_range
			mean_range	:	(float)
							Range (km/s) to move gaussian by 
		Returns
		-------
					Updates values in this class object
		self.gaussians_fit_split	:	(dict)
										Dict containing fits for each individual gaussian
		self.continuum_offset_fit	:	(float)
										Continuum offset output by curve_fit
		self.fl_fit					:	(np.array)
										Flux values for final fit curve
		self.chi_sq 				:	(float)
		self.reduced_chi_sq				Chi squared values for the fit

		'''

		#Add gaussians to a common list for curve_fit to be happy
		argv = []
		for key, val in kwargs['gaussians'].items():
			argv.append(val['amp'])
			argv.append(val['fwhm'])
			argv.append(val['mean'])

		#Setting upper and lower bounds for curve_fit
		min_bound = np.zeros(len(argv)+1)
		max_bound = np.zeros(len(argv)+1)

		for i, arg in enumerate(argv):
			# Bound the amplitude +- percent range
			if i%3 == 0:
				#i+1 since adding continuum offset to beginning of list for profile generation (fitNGaussians)
				#Check if emission
				if arg >= 0:
					min_bound[i+1] = np.multiply(arg, 1 - (kwargs['amp_percent_range']/100.0))
					max_bound[i+1] = np.multiply(arg, 1 + (kwargs['amp_percent_range']/100.0))
				#or absorption
				else:
					min_bound[i+1] = np.multiply(arg, 1 + (kwargs['amp_percent_range']/100.0))
					max_bound[i+1] = np.multiply(arg, 1 - (kwargs['amp_percent_range']/100.0))
			# Bound the FWHM +- percent range
			elif i%3 == 1:
				min_bound[i+1] = np.multiply(arg, 1 - (kwargs['fwhm_percent_range']/100.0))
				max_bound[i+1] = np.multiply(arg, 1 + (kwargs['fwhm_percent_range']/100.0))
			# Bound the mean +- value (km/s)
			elif i%3 == 2:
				min_bound[i+1] = np.subtract(arg, kwargs['mean_range'])
				max_bound[i+1] = np.add(arg, kwargs['mean_range'])

		#Append continuum offset to beginning of initial values
		argv.insert(0, self.continuum_offset)

		#Bound the continuum offset
		if argv[0] > 0:
			min_bound[0] = np.multiply(argv[0], 1 - (kwargs['continuum_offset_percent_range']/100.0))
			max_bound[0] = np.multiply(argv[0], 1 + (kwargs['continuum_offset_percent_range']/100.0))
		else:
			min_bound[0] = np.multiply(argv[0], 1 + (kwargs['continuum_offset_percent_range']/100.0))
			max_bound[0] = np.multiply(argv[0], 1 - (kwargs['continuum_offset_percent_range']/100.0))

		#Try fit to data
		popt, pcov = curve_fit(fitNGaussians, 
								self.vel, self.fl, 
								p0=argv,
			  					bounds=(min_bound, max_bound)
			  					)
		#Save fit data
		self.continuum_offset_fit = popt[0]
		self.continuum_offset_fit_err = np.sqrt(pcov[0,0])
		#Store each seperate gaussian 
		self.gaussians_fit_split = {}
		#len(popt) = 3 * N (gaussians) + 1 (cont. offset)
		for j in range(int((len(popt)-1)/3)):
			#Calculate FWHM values 
			i=j*3
			fwhm = sigma2FWHM(popt[i+2])
			fwhm_err = percentileErrorEquivalence(val_1=popt[i+2], 
												  err_1 = np.sqrt(pcov[i+2,i+2]), 
												  val_2=fwhm)
			fwhm_var = np.power(fwhm_err, 2)

			#Generate dict to store within dict
			key = "Gaussian{}".format(i+1)
			item = {"amp":		popt[i+1],
					"amp_var": 	pcov[i+1,i+1], 
					"amp_err": 	np.sqrt(pcov[i+1,i+1]),
					"sigma":	popt[i+2],
					"sigma_var":pcov[i+2,i+2],
					"sigma_err":np.sqrt(pcov[i+2,i+2]),
					"FWHM": 	fwhm,
					"FWHM_var": fwhm_var,
					"FWHM_err": fwhm_err,
					"mean":		popt[i+3],
					"mean_var": pcov[i+3,i+3],
					"mean_err": np.sqrt(pcov[i+3,i+3]),
					"tuple":	(popt[i+1], popt[i+2], popt[i+3]),
					"tuple_var":(pcov[i+1,i+1], pcov[i+2,i+2], pcov[i+3,i+3]),
			}
			self.gaussians_fit_split[key] = item


		#Generate fit shape
		self.fl_fit = fitNGaussians(self.vel, *popt)
		#Calculate Chi squared values
		self.chiSquared(ngauss=len(kwargs['gaussians']),ncont=1)

	def chiSquared(self, ngauss=1, ncont=1):
		'''
		Calculates error in val_2 if same percentile error as val_1

		Parameters
		----------
		ngauss 	: 	(int)
					Number of gaussians being fit
		ncont	:	(int)
					Number of parameters used for continuum fitting.
					Default 1, for constant offset
		Returns
		-------
		self.chi_sq 				:	(float)
		self.reduced_chi_sq				Chi squared values for the fit

		'''
		obs = self.fl
		exp = self.fl_fit
		var = self.var
		std = self.std

		self.chi_sq = np.sum(np.divide(np.power(obs - exp, 2), np.power(std,2)))
		self.reduced_chi_sq = self.chi_sq / (len(self.wl)-(3*ngauss + ncont))


	def printInfo(self):

		for key, val in self.gaussians_fit_split.items():
			
			print('\n{key}'.format(key=key))
			print('---------')
			print('Amp : {amp:9.5f} +- {err:9.5f}'.format(amp=val["amp"],err=val["amp_err"]))
			print('FWHM: {fwhm:9.5f} +- {err:9.5f}'.format(fwhm=val["FWHM"],err=val["FWHM_err"]))
			print('Mean: {mean:9.5f} +- {err:9.5f}'.format(mean=val["mean"],err=val["mean_err"]))
			print('Continuum Offset: {co:9.5f} +- {err:9.5f}'.format(co=self.continuum_offset_fit, err=self.continuum_offset_fit_err))


		# print("Continuum slope: {m}".format(m=continuum_slope))
		print("Chi squared  : {cs:9.5f}".format(cs=self.chi_sq))
		print("Reduced Chi squared  : {rcs:9.5f}".format(rcs=self.reduced_chi_sq))
		print("\n")


#Changes between FWHM of gaussian and gaussian sigma 
def FWHM2sigma(FWHM):
	return FWHM/(2*np.sqrt(2*np.log(2)))
#Changes between gaussian sigma and FWHM of gaussian
def sigma2FWHM(sigma):
	return (2*np.sqrt(2*np.log(2)))*sigma

def percentileErrorEquivalence(val_1=1, val_2=1, err_1=0):
	'''
	Calculates error in val_2 if same percentile error as val_1

	Parameters
	----------
	val_1	: 	(float) 
					Value with known error
	val_2	:	(float)
					Value with unknown error
	err_1	:	(float)	
					Error of val_1
	Returns
	-------
	err_2	: 	(float) 
					Calculated error in val_2
	'''
	err_2 =  val_2 * err_1 / val_1
	return err_2

def fitNGaussians(x,*argv):
	'''
	Overlays any amount of gaussians

	Parameters
	----------
	x 			: 	(1D np.array) 
					list of wavelengths to compute gaussian at
	offset=0	:	(float)
					Continuum offset to add to gaussians
	gaussians=[]:	(list of tuples) [(amp, sigma, mean), (amp2, sigma2, mean2), ...]	
					List containing tuples defining parameters for each gaussian to be added
	Returns
	-------
	total		 : 	(1D np.array) 
					Computed line profile based on stacking gaussians
	'''
	#Check that args contain 1 + 3N number of variables
	# 1 continuum offset 
	# 3 per gaussian to fit (amp, sigma, mean)
	assert (len(argv)-1)%3 == 0
	#Offset is first argument parsed
	offset = argv[0]
	#Generate a list of all gaussians variables from input
	gaussians = [(argv[i], argv[i+1], argv[i+2]) for i in range(1,len(argv), 3)]


	total = offset
	#For each input gaussian to overlay
	for gaussian in gaussians:
		#Ensure it contains amplitude, FWHM, central value
		assert len(gaussian) == 3
		assert gaussian[1] > 0
		#Define gaussian parameters
		amp = gaussian[0]
		sigma = gaussian[1]
		mean = gaussian[2]
		#Add to total gaussian profile
		total += amp * np.exp(-np.power(x-mean, 2) / (2*np.power(sigma,2)))

	return total