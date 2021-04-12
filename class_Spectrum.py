import numpy as np

#Speed of light in km/s
C = 3E5

class Spectrum:
	def __init__(self, 
				wavelength=[],  		#List of wavelengths
				flux=[], 				#List of flux values at each wavelength
				var=[],					#List of variance for each flux value
				std=[],					#List of sigma for each flux value
				date="",				#Date of observation
				mjd=0, 					#MJD of observation
				mjd_t_peak = 0,			#T_MAX MJD of SN
				instrument='None', 		#Instrument observation
				obj = "",				#Object name
				z=0, 					#Redshift of object
				offset=0, 				#y-offset for spectrum when plotting
				c='black',				#Colour of plot
				):
		self.wl = np.array(wavelength)
		self.fl = np.array(flux)
		self.var = np.array(var)
		self.std = np.array(std)
		self.vel = np.zeros(len(self.wl))

		self.date = date
		self.mjd = mjd
		self.mjd_t_peak = mjd_t_peak
		self.days_from_t_peak = mjd - mjd_t_peak
		self.instrument = instrument
		self.offset = offset
		self.object = obj
		self.c = c
		self.z = z

	#Determines true value of redshift given an initial estimate
	def DetermineRedshift(self, 
						  initial_z=None,		#Initial redshift guess 
						  lam_rest=6564.61, 	#Wavelength to focus on in Angstrom (default Halpha)
						  window=200,			#Range of values to analyse in Angstrom 
						  deredshift=True,		#Deredshift spectrum after calculation
						  ):
		if initial_z == None:
			initial_z = self.z
		#Estimate of redshifted wl given initial redshift value
		est_lam_obs = lam_rest * (initial_z + 1)
		
		#Find index of est_lam_obs
		est_idx_lam_obs = np.searchsorted(self.wl, lam_rest)
		#calculate index of window limits
		min_window_idx = est_idx_lam_obs-window/2
		max_window_idx = est_idx_lam_obs+window/2
		#Find maxima in window around estimated central wl
		idx_fl_obs = min_window_idx + np.argmax(self.fl[min_window_idx:max_window_idx])
		idx_lam_obs = idx_fl_obs
		lam_obs = self.wl[idx_lam_obs]

		#Calculate actual redshift value
		# self.z = lam_obs / (lam_rest * (initial_z + 1)) - 1
		self.z = (lam_obs - lam_rest)/lam_rest
		if deredshift == True:
			self.Deredshift()

	#Shifts wavelengths to rest frame
	def Deredshift(self, z=None):
		if z == None:
			z = self.z
		self.wl = np.array([x/(1+z) for x in self.wl])

	#Plots spectrum with error bars
	def PlotSpectrum(self, ax, c=None, sigma=1, vel=False, alpha=1, name=None, error=True):
		#Calculate error bar size
		if error:
			error_bar = np.multiply(self.std, sigma)
		else:
			error_bar = np.zeros(len(self.wl))
		#If colour not specified, use predefined colour
		if c == None:
			c = self.c
		#Label for the line
		if name == None:
			name = self.object
		#Offset flux values
		flux = np.add(self.fl, self.offset)
		#Plot spectrum
		if vel == False:
			ax.plot(self.wl, flux, color=c, alpha=alpha, linestyle='-')
			#Plot error bars
			ax.fill_between(self.wl, np.add(flux, error_bar), np.subtract(flux, error_bar), color=c, alpha=0.5*alpha)

			ax.text(self.wl[-1] - 100, flux[-1] + 0.1,  #Position of text
				"{}".format(name),
							color=c,					#Formatting text
							)
		
		else:
			ax.plot(self.vel, flux, color=c, alpha=alpha, linestyle=':')
			#Plot error bars
			ax.fill_between(self.vel, np.add(flux, error_bar), np.subtract(flux, error_bar), color=c, alpha=0.5*alpha)

			ax.text(self.vel[-1] - 100, flux[-1] + 0.1, #Position of text
				"{}".format(name),
							color=c,						#Formatting text
							)

	def SaveSpectrum(self, filename):
		np.savetxt(filename, np.column_stack((self.wl, self.fl, self.std)), header="wavelength,flux,err", delimiter=',')


	def SubtractSpectrum(self, spectrum):
		self.fl = np.subtract(self.fl, spectrum.fl)
		self.std = np.add(self.std, spectrum.std)
		self.var = np.add(self.var, spectrum.var)


	def rmNegative(self, cutoff=None):
		if cutoff != None:
			for i,f in enumerate(self.fl):
				if f < cutoff:
					self.fl[i] = np.nan

	#Normalise spectrum between norm_min and norm_max
	def Normalise(self, curr_min = None, curr_max = None, norm_min=0, norm_max=1, norm_factor=None, ignore_range=None):

		#Remove values from normalisation
		if ignore_range != None:
			orig_fl = self.fl
			orig_wl = self.wl
			orig_var = self.var
			orig_std = self.std
			#Cut out unwanted region
			for r in ignore_range:
				min_wl = r[0]
				max_wl = r[1]
				min_idx = np.searchsorted(self.wl, min_wl)
				max_idx = np.searchsorted(self.wl, max_wl)
				self.wl = np.append(self.wl[:min_idx], self.wl[max_idx:])
				self.fl = np.append(self.fl[:min_idx], self.fl[max_idx:])
				self.var = np.append(self.var[:min_idx], self.var[max_idx:])
				self.std = np.append(self.std[:min_idx], self.std[max_idx:])

		#If not set, normalise all spectrum values
		if curr_min == None:
			curr_min = np.nanmin(self.fl)
		if curr_max == None:
			curr_max = np.nanmax(self.fl)

		#If not ignoring anything, set to default
		if ignore_range != None:
			self.fl = orig_fl
			self.wl = orig_wl
			self.var = orig_var
			self.std = orig_std
		
		#Set min to 0
		flux = np.subtract(self.fl, curr_min)
		#Calculate a normalisation factor
		if norm_factor == None:
			norm_factor = np.divide(norm_max - norm_min, curr_max - curr_min)
		#Normalise 
		flux = np.multiply(flux,norm_factor)
		#Add floor value
		self.fl = np.add(flux, norm_min)
		#Calculate what happens to errors
		std_factor = (norm_max - norm_min)/(curr_max - curr_min)
		var_factor = std_factor ** 2
		
		self.std = np.multiply(self.std, std_factor)
		self.var = np.multiply(self.var, var_factor)

		#Output normalisation factor
		return norm_factor


	def Scale(self, factor=1):
		self.fl = np.multiply(self.fl, factor)
		self.std = np.multiply(self.std, factor)
		self.var = np.multiply(self.var, np.power(factor,2))


	#Trim wavelengths to within specified range
	def TrimWL(self,min_wl=0, max_wl=10000):
		i_min = np.searchsorted(self.wl, min_wl, side='left')
		i_max = np.searchsorted(self.wl, max_wl, side='left')
		self.wl = self.wl[i_min:i_max]
		self.fl = self.fl[i_min:i_max]
		self.vel = self.vel[i_min:i_max]
		self.std = self.std[i_min:i_max]
		self.var = self.var[i_min:i_max]

	#Trim velocities to within specified range
	def TrimVel(self,min_vel=-1000, max_vel=1000):
		i_min = np.searchsorted(self.vel, min_vel, side='left')
		i_max = np.searchsorted(self.vel, max_vel, side='left')
		self.wl = self.wl[i_min:i_max]
		self.fl = self.fl[i_min:i_max]
		self.vel = self.vel[i_min:i_max]
		self.std = self.std[i_min:i_max]
		self.var = self.var[i_min:i_max]

	#
	def wl2vel(self, centre=6564.5377):
		# wl = np.subtract(wl, centre)
		self.vel = np.multiply(np.divide(np.subtract(self.wl,centre), centre), C)

	#Remove a wavelength range from the data
	def rmWL(self, min_wl, max_wl):
		#Find the index of these values
		min_idx = np.searchsorted(self.wl, min_wl)
		max_idx = np.searchsorted(self.wl, max_wl)
		#Remove from data
		self.wl = np.append(self.wl[:min_idx], self.wl[max_idx:])
		self.fl = np.append(self.fl[:min_idx], self.fl[max_idx:])
		self.vel = np.append(self.vel[:min_idx], self.vel[max_idx:])
		self.std = np.append(self.std[:min_idx], self.std[max_idx:])
		self.var = np.append(self.var[:min_idx], self.var[max_idx:])