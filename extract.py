from class_Line import *
from class_Spectrum import *
from fn_spaxelSelection import *

#Numpy
import numpy as np
import pandas as pd
#Plotting tools
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib as mpl
#FITS manipulation
from astropy.io import fits
#Create copy of class instances
from copy import deepcopy

#Ordered dictionary
import collections as coll

#Region around line to analyse in Angstrom
window = 200
#How much to raise each line up by for plots
PLT_SHIFT = 0.7
# PLT_SHIFT = 0

#Spectral lines to check for
#HAVEN'T IMPLEMENTED
LINE_H_I    = [6564.61, 4862.68, 4341.68, 4102.89]

LINE_He_I   = [5875.624,7065.196,7281.349,7816.136, ]
LINE_He_II  = [8236.790]

LINE_C_II   = [7236.420]
LINE_C_III  = [4647.420,4650.250,4651.470,5695.920,]
LINE_C_IV   = [5801.330, 5811.980]

LINE_N_II   = [6549.86,6585.27]

LINE_O_II   = [3727.092,3729.875]
LINE_O_III  = [1665.85,4364.436,4932.603,4960.295,5008.240]

LINE_S_II   = [4072.3,6718.29,6732.67]

LINE_Si = []
    
    

#Load in data to manipulate
def loadFITS(filename):
    '''
    Reads in data from finalised WiFeS P11 file.
    Can be used for any data cube where 
    HDU 0 is science data
    HDU 1 is variance data
    HDU 2 is data quality flag

    Parameters
    ----------
    filename :  (str) File to read in, inclusive of file path

    Returns
    -------
    sci : (np.array) Science data cube
    var : (np.array) Variance data cube
    dq  : (np.array) Data quality date cube
    '''
    hdul = fits.open(filename)
    sci = hdul[0]
    var = hdul[1]
    dq  = hdul[2]

    return sci, var, dq
#Average the image data across both WiFeS arms
def aveImage(*datacubes):
    '''
    Averages a data cube over all wavelengths to produce 2D image

    Parameters
    ----------
    *datacubes  :   (List of 3D np.array) 
                    Science spaxels to image

    Returns
    -------
    ave :   (2D np.array) 
            Image of science data
    '''
    #Set total = 0 for array in size of image
    total = np.zeros_like(datacubes[0][0])
    #For each data cube to average
    for cube in datacubes:
        #Assert dimensions are correct
        assert(cube.shape[1] == total.shape[0])
        assert(cube.shape[2] == total.shape[1])
        #Average the flux across wavelength axis
        ave = np.mean(cube, axis=0)
        #Add to total
        total = np.add(total, ave)

    #Make it an average instead of total count
    ave = np.divide(total, len(datacubes))

    return ave
#Calculate the flux values at each wavelength step, weighted by variance
def calcFlux(sci, var, save_x, save_y, sub_x, sub_y):
    '''
    Takes in user selected range of spaxels and averages flux for each spaxel per wavelength.
    Subtracts spaxels in another user selected region for sky/host subtraction
    Weights flux values by variance of spaxel (i.e. higher variance = less weight)

    Parameters
    ----------
    sci     :   (3D np.array) 
                Science data cube

    var     :   (3D np.array)
                Variance data cube

    save_x, :   (dict){'start':(int), 'end':(int)}
    save_y      Coordinates to average saved spaxels across

    sub_x,  :   (dict){'start':(int), 'end':(int)}
    sub_y       Coordinates to average subtracted spaxels across

    Returns
    -------
    fl :    (2D np.array) 
            Spectrum of selected range
    '''
    #Extracts average spectrum in section to save
    save_sci = sci.data[:, save_y['start']:save_y['end'], save_x['start']:save_x['end']]
    save_var = var.data[:, save_y['start']:save_y['end'], save_x['start']:save_x['end']]
    #Extracts average spectrum in section to subtract
    sub_sci  = sci.data[:, sub_y['start']:sub_y['end'], sub_x['start']:sub_x['end']]
    sub_var  = var.data[:, sub_y['start']:sub_y['end'], sub_x['start']:sub_x['end']]
    #Calculates the weighted average spectrum across selection range
    fl = [
            np.average(save_sci[i], weights=np.reciprocal(save_var[i])) - 
            np.average(sub_sci[i], weights=np.reciprocal(sub_var[i])) 
            for i in range(save_sci.shape[0])
        ]
    # fl = [
    #       np.sum(save_sci[i]) - 
    #       np.sum(sub_sci[i]) 
    #       for i in range(save_sci.shape[0])
    #   ]
    return fl
#Calculate the variance for each flux value
def calcVar(var, save_x, save_y, sub_x, sub_y):
    '''
    Calculates variance in flux values across selected region

    Parameters
    ----------
    var     :   (3D np.array)
                Variance data cube

    save_x, :   (dict){'start':(int), 'end':(int)}
    save_y      Coordinates to average saved spaxels across

    sub_x,  :   (dict){'start':(int), 'end':(int)}
    sub_y       Coordinates to average subtracted spaxels across

    Returns
    -------
    err :   (2D np.array) 
            Added error of spaxels in selected ranges.
    '''
    #Cut out relevant regions
    save_var = var.data[:, save_y['start']:save_y['end'], save_x['start']:save_x['end']]
    sub_var  = var.data[:, sub_y['start']:sub_y['end'], sub_x['start']:sub_x['end']]
    #Calculate standard error of weighted mean
    save_err = np.reciprocal(np.sum(np.reciprocal(save_var), axis=(1,2)))
    sub_err = np.reciprocal(np.sum(np.reciprocal(sub_var), axis=(1,2)))
    #Add errors of two sections
    return save_err + sub_err
#Calculate an array of each wavelength
def calcWavelength(sci):
    '''
    Calculates an array of wavelengths for each flux value to correspond to, 
    since not included by default in data

    Parameters
    ----------
    sci     :   (3D np.array) 
                Science data cube

    Returns
    -------
    wl :    (2D np.array) 
            Array for each wavelength in data cube
    '''
    initial = sci.header['CRVAL3']
    step = sci.header['CDELT3']
    num = len(sci.data)
    final = initial + step * num

    return np.arange(initial, final, step)
#Combine two spectra (e.g. red and blue arms of WiFeS)
def combineSpectra(b_fl, b_var, b_wl, r_fl, r_var, r_wl):
    '''
    Combines blue and red arms of WiFeS spectra. Calculates an overlap region, then 
    adjusts each spectrum to meet in middle.

    Parameters
    ----------
    b_fl,   :   (1D np.array) 
    r_fl        Flux values for each arm
    b_var,  :   (1D np.array)
    r_var       Variance values for each arm
    b_wl,   :   (1D np.array)
    r_wl        Wavelength values for each arm

    Returns
    -------
    comb_fl :   (1D np.array)
                Combined flux array
    comb_var :  (1D np.array)
                Combined variance array
    comb_wl :   (1D np.array)
                Combined wavelength array
    '''

    #Determine what extreme ends of filters wavelengths are
    max_b = max(b_wl)
    min_r = min(r_wl)

    #Determine how many entries overlap in both 
    b_overlap = len([i for i in b_wl if i > min_r])
    r_overlap = len([i for i in r_wl if i < max_b])
    
    #Calculate the average flux within this range
    b_ave = np.mean(b_fl[-b_overlap:])
    r_ave = np.mean(r_fl[:r_overlap])
    
    #Calculate the difference between the two in the overlapping region
    # r_offset = (b_ave - r_ave)/2
    # b_offset = (r_ave - b_ave)/2
    ############################################
    r_offset = 0
    b_offset = 0
    ############################################
    
    #Shift spectra to meet at average
    r_fl = [x + r_offset for x in r_fl]
    b_fl = [x + b_offset for x in b_fl]
    
    #Combine lists
    comb_wl = np.concatenate((b_wl, r_wl))
    comb_var= np.concatenate((b_var, r_var))
    comb_fl = np.concatenate((b_fl, r_fl))

    #Zip them together
    zipped = list(zip(comb_wl, comb_fl, comb_var))
    #Sort according to wavelength
    zipped.sort(key = lambda x: x[0])
    
    #Recreate combined lists, now sorted to wavelength
    unzipped = [list(x) for x in zip(*zipped)]
    comb_wl = unzipped[0]
    comb_fl = unzipped[1]
    comb_var= unzipped[2]
    
    return comb_fl, comb_var, comb_wl 
#Takes in blue and red filenames, outputs Spectrum object
def processData(blue_file, red_file, z=0, c='black', mjd_t_peak=0, instrument="WiFeS", obj="SN", offset=0, day=''):
    '''
    Imports data cubes, extracts important info out of FITS headers,
    extracts spectra out of user selected region in data cube,
    creates a Spectrum object

    Parameters
    ----------
    blue_file,          :   (str)
    red_file                File names of WiFeS data cubes to combine and extract info from
    
    z=0                 :   (float)
                            Redshift of object
    c='black'           :   (str)
                            Colour to plot spectrum in
    mjd_t_peak=0        :   (int)
                            MJD of time of SN max
    instrument="wifes"  :   (str)
                            Name of instrument
    obj="SN"            :   (str)
                            Name of object
    offset=0            :   (int)
                            How much to shift spectrum up in plotting so not all spectra are overlaid
    day=''              :   (str)
                            Date of observations

    Returns
    -------
    processed_spectrum  :   (Spectrum object)
                            Filled out Spectrum object from class_Spectrum.py
    '''

    #Load in data
    b_sci, b_var, _ = loadFITS(blue_file)
    r_sci, r_var, _ = loadFITS(red_file)
    
    #Extract MJD out of header
    mjd = b_sci.header['MJD-OBS']
    date = b_sci.header['DATE-OBS']
    bad_obj_names = ["", "TOO"]
    if b_sci.header['NOTES'].upper() not in bad_obj_names:
        obj = b_sci.header['NOTES']

    #Generate image for user to see
    ave_image = aveImage(r_sci.data, b_sci.data)

    #Get user selection of pixels to analyse/reduce
    save_x, save_y = select_spaxel(ave_image, date=date,)
    sub_x, sub_y = select_spaxel(ave_image, date=date,
                                rect  = (save_x['start'], save_y['start']),
                                width =  save_x['end']-save_x['start'],
                                height=  save_y['end']-save_y['start'],
                                )

    #Reset 
    start = [None, None]
    end = [None, None]


    #Calculate spectrum for selected values
    b_fl = calcFlux(b_sci, b_var, save_x, save_y, sub_x, sub_y)
    b_var= calcVar(b_var, save_x, save_y, sub_x, sub_y)
    b_wl = calcWavelength(b_sci)

    r_fl = calcFlux(r_sci, r_var, save_x, save_y, sub_x, sub_y)
    r_var= calcVar(r_var, save_x, save_y, sub_x, sub_y)
    r_wl = calcWavelength(r_sci)

    #Combine spectra into single array
    fl, var, wl = combineSpectra(b_fl, b_var, b_wl, r_fl, r_var, r_wl)
    std = np.sqrt(var)

    #Create spectrum object
    processed_spectrum = Spectrum(wavelength=wl,
                                    flux=fl,
                                    var=var,
                                    std=std,
                                    date=date,
                                    mjd=mjd,
                                    mjd_t_peak=mjd_t_peak,
                                    instrument=instrument,
                                    obj=obj,
                                    z=z,
                                    offset=offset,
                                    c=c,
                                    )

    return processed_spectrum
#Returns order of magnitude of a value
def magnitude(value):
    '''Returns order of magnitude of a float value'''
    return int(np.log10(np.absolute(value)))

if __name__ == "__main__":

    days = {}
    #2019com
    # days['20190401_G-S']  = {'b_file': 'cubes/T2m3wb-20190401.174404-0052.p11.fits', 'r_file': 'cubes/T2m3wr-20190401.180513-0053.p11.fits', 'colour': 'red'}
    # days['20190401_SN-G'] = {'b_file': 'cubes/T2m3wb-20190401.174404-0052.p11.fits', 'r_file': 'cubes/T2m3wr-20190401.180513-0053.p11.fits', 'colour': 'green'}
    # days['20190401_SN-S'] = {'b_file': 'cubes/T2m3wb-20190401.174404-0052.p11.fits', 'r_file': 'cubes/T2m3wr-20190401.180513-0053.p11.fits', 'colour': 'black'}
    # days['20190401'] = {'b_file': 'cubes/T2m3wb-20190401.174404-0052.p11.fits', 'r_file': 'cubes/T2m3wr-20190401.180513-0053.p11.fits', 'colour': 'green'}
    # days['20190402'] = {'b_file': 'cubes/T2m3wb-20190402.174823-0059.p11.fits', 'r_file': 'cubes/T2m3wr-20190402.174823-0059.p11.fits', 'colour': 'blue'}
    # days['20190403'] = {'b_file': 'cubes/T2m3wb-20190403.173753-0066.p11.fits', 'r_file': 'cubes/T2m3wr-20190403.173753-0066.p11.fits', 'colour': 'purple'}
    # days['20190404'] = {'b_file': 'cubes/T2m3wb-20190404.152853-0001.p11.fits', 'r_file': 'cubes/T2m3wr-20190404.152853-0001.p11.fits', 'colour': 'cyan'}
    # days['20190423'] = {'b_file': 'cubes/T2m3wb-20190423.180348-0022.p11.fits', 'r_file': 'cubes/T2m3wr-20190423.173728-0020.p11.fits', 'colour': 'brown'}
    # days['20190426'] = {'b_file': 'cubes/T2m3wb-20190426.174216-0510.p11.fits', 'r_file': 'cubes/T2m3wr-20190426.172117-0509.p11.fits', 'colour': 'orange'}
    days['20190427'] = {'b_file': 'cubes/T2m3wb-20190427.131154-0104.p11.fits', 'r_file': 'cubes/T2m3wr-20190427.135355-0106.p11.fits', 'colour': 'red'}
    # days['20190919'] = {'b_file': 'cubes/IC_4712_b_20190919.fits', 'r_file': 'cubes/IC_4712_r_20190919.fits', 'colour':'black'}
    # days['20190922'] = {'b_file': 'cubes/IC_4712_b_20190922.fits', 'r_file': 'cubes/IC_4712_r_20190922.fits', 'colour':'blue'}
    # days['20191109'] = {'b_file': 'cubes/IC_4712_b_20191109.fits', 'r_file': 'cubes/IC_4712_r_20191109.fits', 'colour':'black'}
    # days['Centre']  = {'b_file': 'cubes/IC_4712_b_20190922.fits', 'r_file': 'cubes/IC_4712_r_20190922.fits', 'colour':'red'} #20190922
    # days['Local'] = {'b_file': 'cubes/IC_4712_b_20190922.fits', 'r_file': 'cubes/IC_4712_r_20190922.fits', 'colour':'blue'}  #20190922


    #2019qiz
    # days['19_small']        = {'b_file': '2019qiz/19/T2m3wb-20190919.165814-0100.p11.fits',      'r_file': '2019qiz/19/T2m3wr-20190919.165814-0100.p11.fits', 'colour': 'green'}
    # days['19_large']        = {'b_file': '2019qiz/19/T2m3wb-20190919.165814-0100.p11.fits',      'r_file': '2019qiz/19/T2m3wr-20190919.165814-0100.p11.fits', 'colour': 'green'}
    # days['19_small_std1'] = {'b_file': '2019qiz/19/stds/T2m3wb-20190919.154037-0009.p11.fits', 'r_file': '2019qiz/19/stds/T2m3wr-20190919.154037-0009.p11.fits', 'colour': 'green'}
    # days['19_large_std1'] = {'b_file': '2019qiz/19/stds/T2m3wb-20190919.154037-0009.p11.fits', 'r_file': '2019qiz/19/stds/T2m3wr-20190919.154037-0009.p11.fits', 'colour': 'green'}
    # days['19_small_std2'] = {'b_file': '2019qiz/19/stds/T2m3wb-20190919.155924-0012.p11.fits', 'r_file': '2019qiz/19/stds/T2m3wr-20190919.155924-0012.p11.fits', 'colour': 'green'}
    # days['19_large_std2'] = {'b_file': '2019qiz/19/stds/T2m3wb-20190919.155924-0012.p11.fits', 'r_file': '2019qiz/19/stds/T2m3wr-20190919.155924-0012.p11.fits', 'colour': 'green'}
    
    # days['22']        = {'b_file': '2019qiz/22/T2m3wb-20190922.164224-0058.p11.fits',      'r_file': '2019qiz/22/T2m3wr-20190922.164224-0058.p11.fits', 'colour': 'blue'}
    # days['22_std1'] = {'b_file': '2019qiz/22/stds/T2m3wb-20190922.084757-0034.p11.fits', 'r_file': '2019qiz/22/stds/T2m3wr-20190922.084757-0034.p11.fits', 'colour': 'blue'}
    # days['22_small_std1'] = {'b_file': '2019qiz/22/stds/T2m3wb-20190922.084757-0034.p11.fits', 'r_file': '2019qiz/22/stds/T2m3wr-20190922.084757-0034.p11.fits', 'colour': 'blue'}
    # days['22_large_std2'] = {'b_file': '2019qiz/22/stds/T2m3wb-20190922.085450-0037.p11.fits', 'r_file': '2019qiz/22/stds/T2m3wr-20190922.085450-0037.p11.fits', 'colour': 'blue'}

    SN2019com = {}
    #For each day to analyse
    for i, key in enumerate(sorted(days.keys())):
        info = days[key]
        print(key)
        #Extract spectrum, merge blue and red arms
        SN2019com[key] = processData(info['b_file'], info['r_file'], c=info['colour'], mjd_t_peak=58590, obj="SN2019com", offset=-i*PLT_SHIFT)
        # SN2019com[key].DetermineRedshift(lam_rest=LINE_H_I[0],
        #                               initial_z=0.0124,
        #                               window=window,
        #                               deredshift=True,
        #                               )


    # files = [ 
    #           # 'spectra/-7.0.dat',   
    #           # 'spectra/+17.0.dat'   
    #           # 'spectra/+180.1.dat'  
    #           ]
    # SN2009ip = {}
    # for file in files:
    #   day = file[8:-6]
    #   wl, fl = np.loadtxt(file, unpack=True)

    #   SN2009ip[day] = Spectrum(wavelength=wl,
    #                               flux=fl,
    #                               date=r't_max {}'.format(day),
    #                               mjd=0,
    #                               mjd_t_peak=0,
    #                               instrument='unknown',
    #                               obj='SN2009ip',
    #                               z=0.005944,
    #                               offset=1,
    #                               c='black',
    #                               )
    #   SN2009ip[day].DetermineRedshift(lam_rest=LINE_H_I[0],
    #                           initial_z=0.005944,
    #                           window=window,
    #                           deredshift=True,
    #                           )


    # gaussians = {}
    # gaussians['G1'] = {'amp': 0.10, 'fwhm': FWHM2sigma(10400), 'mean':0}
    # gaussians['G2'] = {'amp': 0.90, 'fwhm': FWHM2sigma(230)  , 'mean':0}
    # gaussians['G3'] = {'amp':-0.10, 'fwhm': FWHM2sigma(200)  , 'mean':800}
    # gaussians['G4'] = {'amp': 0.40, 'fwhm': FWHM2sigma(1500) , 'mean':0}
    
    tempSN2 = {}

    fig, ax = plt.subplots()
    #Find H_alpha
    for i, day in enumerate(days):
        #Create a deepcopy so can overwrite each loop
        tempSN = deepcopy(SN2019com[day])

        # Trim around spectral line, convert to velocity space
        # tempSN.TrimWL(min_wl=LINE_H_I[0]-window/2, max_wl=LINE_H_I[0]+window/2)
        tempSN.TrimWL(min_wl=3800, max_wl=7500)
        # tempSN.wl2vel(centre=LINE_H_I[0])
        tempSN.Normalise(ignore_range=[[5300, 5600], [8000,10000]])
        tempSN.SaveSpectrum("2019com/spectra/{}.csv".format(day))

        #Normalise flux
        # tempSN.Scale(factor=1E16)

        # #Create a line object
        # SN2019com_Ha = SNLine(wl=tempSN.wl, 
        #                     fl=tempSN.fl, 
        #                     vel=tempSN.vel, 
        #                     var=tempSN.var, 
        #                     std=tempSN.std, 
        #                     colour=tempSN.c, 
        #                     date=day
        #                   )
        # #Fit gaussians to it
        # SN2019com_Ha.fitCurve(gaussians=gaussians,
        #                     amp_percent_range=100, #Bounds (percent)
        #                     fwhm_percent_range=20, #Bounds (percent)
        #                     continuum_offset_percent_range=1, #Bounds (percent)
        #                     mean_range=200, #Bounds (km/s)
        #                     )


        # tempSN.Scale(factor=1E-16)
        # SN2019com_Ha.Scale(factor=1E-16)

        #Trim plot to show just Halpha
        # tempSN.TrimVel(min_vel=-2000, max_vel=2000)
        # SN2019com_Ha.TrimVel(min_vel=-2000, max_vel=2000)

        #Plot original data
        tempSN.PlotSpectrum(ax, sigma=1, vel=False, alpha=0.8, name=day)
        # tempSN.SaveSpectrum("2019qiz/spectra/NORMALISED_{}_centre-galaxy.csv".format(day))
        #Plot fit data
        # y = np.add(SN2019com_Ha.fl_fit, tempSN.offset)
        # plt.plot(SN2019com_Ha.wl, y, color=SN2019com_Ha.colour, linestyle='-')

        # SN2019com_Ha.printInfo()


        # tempSN2[day] = deepcopy(tempSN)

    # for i, day in enumerate(SN2009ip):
    #   print(SN2009ip[day].fl)
    #   #Create a deepcopy so can overwrite each loop
    #   tempSN = deepcopy(SN2009ip[day])
    #   # Trim around spectral line, convert to velocity space
    #   tempSN.TrimWL(min_wl=3900, max_wl=7000)
    #   #Normalise flux
    #   tempSN.Normalise(ignore_range=[[5300, 5600], [8000,10000]])
    #   #Plot original data
    #   tempSN.PlotSpectrum(ax, sigma=1, vel=False, alpha=0.8, name=day, error=False)


    #Write extracted spectrum to file
    # for day in days:

    #   data = np.transpose([SN2019com[day].wl, SN2019com[day].fl])
    #   np.savetxt('spectra/2019com_201904{}.dat'.format(day), data)

    ax.set_xlabel(r'Wavelength ($\AA$)',fontsize=16,family='serif')
    # ax.set_xlabel(r'Velocity (km $\mathrm{s}^{-1}$)',fontsize=16,family='serif')
    ax.set_ylabel(r'Normalised Flux + Offset',fontsize=16,family='serif')
    
    #Add legend
    # data = mlines.Line2D([],[],color='black',marker='.', linestyle='none', label='Data')
    # fit = mlines.Line2D([],[],color='black', linestyle='-', label='Fit')
    # ax.legend(handles=[data, fit],loc=1)

    #Add emission markers
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    top_offset = np.subtract(ylim[1], ylim[0])*0.1
    top = ylim[1] - top_offset
    side_offset = np.subtract(xlim[1], xlim[0])*0.03

    # ax.axvline(x=LINE_H_I[0],  color='black', linestyle='--', alpha=0.8)
    # ax.text(LINE_H_I[0] + side_offset, top, r'H$\alpha$')
    # ax.axvline(x=LINE_N_II[0], color='black', linestyle='--', alpha=0.8)
    # ax.text(LINE_N_II[0] + side_offset, top, 'N II')
    # ax.axvline(x=LINE_N_II[1], color='black', linestyle='--', alpha=0.8)
    # ax.text(LINE_N_II[1] + side_offset, top, 'N II')

    #Format ticks
    ax.minorticks_on()
    ax.tick_params(axis='both',
                which='major',
                direction='in',
                length=5,
                width=1,
                color='black',
                top=True,
                right=True,
                labelsize=12,
                )
    ax.tick_params(axis='both',
                which='minor',
                direction='in',
                length=2.5,
                color='black',
                top=True,
                right=True,
                labelsize=12,
                )
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    plt.show()
    plt.close()
