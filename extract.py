from class_Line import *
from class_Spectrum import *
from fn_spaxelSelection import *

import getopt
import sys

#Numpy
import numpy as np
#Plotting tools
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib as mpl
#FITS manipulation
from astropy.io import fits



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
    
#Region around line to analyse in Angstrom
velocity_window = 20000

telluric_noise = [[5300, 5600], [8000,10000]]
trimmin = 3800
trimmax = 7500

#Default values for command line arguments
plot_spectrum = True
save_spectrum = False
normalise_spectrum = False
trim_spectrum = False
velocity_space = False
velocity_centre = LINE_H_I[0]
calc_gaussian = False
colour = 'black'
label = 'unlabelled'
redshift = 0
spectral_lines = False
b_file = ''
r_file = ''
fwhm=10000
    

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
    r_offset = (b_ave - r_ave)/2
    b_offset = (r_ave - b_ave)/2
    ############################################
    #r_offset = 0
    #b_offset = 0
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
    save_x, save_y = select_spaxel(ave_image, title='Select transient spaxels to add',)
    sub_x, sub_y = select_spaxel(ave_image, title='Select sky spaxels to subtract',
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
#Reads command line arguments
def read_cla(argv):
    '''
    Reads in command line arguments, and sets flags for 
    remainder of code to work through

    Parameters
    ----------
    argv,          :   (list) argument vector, lists all things after 'python extract.py'                

    Returns
    -------
    options        :   (Various)
                       Boolean values for each flag, text for object label
    filenames      :   (string)
                       File names of red and blue side
    '''

    global plot_spectrum
    global save_spectrum
    global normalise_spectrum
    global velocity_space
    global velocity_centre
    global calc_gaussian
    global colour
    global label
    global redshift
    global trim_spectrum
    global trimmin
    global trimmax
    global b_file
    global r_file
    global spectral_lines
    global velocity_window
    global fwhm


    options, remainder = getopt.getopt(argv, 'c:l:r:hnsvgt', ['help',
                                                            'noplot',
                                                            'save-spectrum',
                                                            'normalise-spectrum',
                                                            'velocity-space',
                                                            'velocity-centre=',
                                                            'fit-gaussian',
                                                            'colour=',
                                                            'color=',
                                                            'label=',
                                                            'rest-frame=',
                                                            'trim-spectrum',
                                                            'trim-min=',
                                                            'trim-max=',
                                                            'spectral-lines',
                                                            'v-window='
                                                        ])

    for opt, arg in options:
        if opt in ('-h', '--help'):
            print(
                '''
                Usage: python extract.py [OPTIONS] blue_p11_file.fits red_p11_file.fits

                Reads in p11 files from PyWiFeS and creates a spectrum based off selected pixels.
                Two windows will pop up consecutively. In the first one, select the target's spaxels.
                In the second one, select some clear region of sky to subtract from the spectrum.
                If all goes well, then a sky subtracted spectrum will pop out

                OPTIONS:
                -c (or --colour)
                    Colour to plot

                -g (or --fit-gaussian)
                    Attempts to fit a gaussian to to the line profile

                -h (or --help)
                    List out all command line options

                -l (or --label, default inferred from filename)
                    Name of object, defaults to date and unique ID of WiFeS file name
               
                -n (or --normalise-spectrum default = False)
                    Normalise spectrum for better plotting comparison
                
                -s (or --save-spectrum, default = False)
                    Save spectrum as CSV, with columns 'wavelength', 'flux', 'std dev'
                
                -t (or --trim-spectrum, default = False)
                    Enables trimming the wavelength range of the spectrum. To be used with --trim-min and --trim-max
                
                -v (or --velocity space, default = False)
                    Converts x-axis into velocity space. To be used with --velocity-centre and --fit-gaussian
                
                --velocity-centre (default = 6564.61 Angstrom = H alpha)
                    Central wavelength to define as '0' velocity, in Angstrom
                
                -z (or --redshift, default = 0)
                    Sets a redshift for redshift correction. If not entered, assumes target already in rest frame
                
                --trim-min (default = 3800)
                    Wavelength at which all shorter wavelengths are discarded (in Angstrom)
                
                --trim-max (default = 7500)
                    Wavelength at which all longer wavelengths are discarded (in Angstrom)
                
                --noplot (default = False)
                    Turns off plotting
                
                --spectral-lines (default = False)
                    Mark common spectral lines on plot

                --fwhm (default = 10000)
                    Width of gaussian to fit if --fit-gaussian enabled (in km/s) 
                
                --vel-window (default = 20000)
                    x limits if plotting in velocity space
                '''
                )
            exit()

        elif opt in ('-c','--colour', '--color'):
            colour = arg
            
        elif opt in ('-s', '--save-spectrum'):
            save_spectrum = True

        elif opt in ('-n','--normalise-spectrum'):
            normalise_spectrum = True

        elif opt in ('-v','--velocity-space'):
            velocity_space = True

        elif opt in ('--velocity-centre'):
            velocity_centre = float(arg)

        elif opt in ('-g', '--fit-gaussian'):
            calc_gaussian = True


        elif opt in ('-z','--redshift'):
            redshift = float(arg)

        elif opt in ('-t', '--trim-spectrum'):
            trim_spectrum = True
        elif opt in ('--trim-min'):
            trimmin = float(arg)
        elif opt in ('--trim-max'):
            trimmax = float(arg)

        elif opt in ('--label'):
            label = arg

        elif opt in ('--noplot'):
            plot_spectrum = False

        elif opt in ('--spectral-lines'):
            spectral_lines = True

        elif opt in ('--vel-window'):
            velocity_window = float(arg)

        elif opt in ('--fwhm'):
            fwhm = float(arg)

    # If incorrect number of files parsed in
    if len(remainder) != 2:
        print("Usage: python extract.py [OPTIONS] blue_p11_file.fits red_p11_file.fits")
        exit()

    else:
        # If no label specified in command line arguments
        if label == 'unlabelled':
            #e.g. T2m3wb-20190919.165814-0100.p11
            #extract '20190919.165814' = 'date.uID'
            label = remainder[0].split('-')[1]
        #Attempts to correctly order files if red was parsed before blue accidentally
        for filename in remainder:
            if 'T2m3wb' in filename:
                b_file = filename
            elif 'T2m3wr' in filename:
                r_file = filename


    # return plot_spectrum, save_spectrum, normalise_spectrum, velocity_space, velocity_centre, calc_gaussian, colour, label, redshift, remainder[0], remainder[1]

if __name__ == "__main__":

    #Read in the command line arguments
    argv = sys.argv[1:]
    read_cla(argv)

    # Read in data from P11's
    transient = processData(b_file, r_file, c=colour, obj=label, z=redshift)
    transient.Deredshift()





    if trim_spectrum == True:
        transient.TrimWL(min_wl=trimmin, max_wl=trimmax)

    if velocity_space == True:
        transient.Normalise(ignore_range=telluric_noise)  
        transient.wl2vel(centre=velocity_centre)
        
        if calc_gaussian == True:
            initial_guess = {'amp': 0.90, 'fwhm': FWHM2sigma(fwhm)  , 'mean':0}
            transient_spectral_line = SNLine(wl=transient.wl, 
                                            fl=transient.fl, 
                                            vel=transient.vel, 
                                            var=transient.var, 
                                            std=transient.std, 
                                            colour=transient.c, 
                                          )
            transient_spectral_line.fitCurve(gaussians={'G1':initial_guess},
                                            amp_percent_range=100, #Bounds (percent)
                                            fwhm_percent_range=20, #Bounds (percent)
                                            continuum_offset_percent_range=1, #Bounds (percent)
                                            mean_range=200, #Bounds (km/s)
                                            )
            transient_spectral_line.printInfo()


    
    if normalise_spectrum == True:
        transient.Normalise(ignore_range=telluric_noise)    

    if save_spectrum == True:
        transient.SaveSpectrum('{}.csv'.format(label))

    #If user wants a plot
    if plot_spectrum == True:
        #Set up axes
        fig, ax = plt.subplots()

        if velocity_space == True:
            # Set axis labels
            ax.set_xlabel(r'Velocity (km $\mathrm{s}^{-1}$)',fontsize=16,family='serif')
            transient.PlotSpectrum(ax, sigma=1, vel=True, alpha=0.8)
            ax.set_xlim(-velocity_window/2, velocity_window/2)

            if calc_gaussian == True:
                ax.plot(transient_spectral_line.vel, transient_spectral_line.fl_fit, color='red', linestyle='--')
        else:
            # Set axis labels
            ax.set_xlabel(r'Wavelength ($\AA$)',fontsize=16,family='serif')
            transient.PlotSpectrum(ax, sigma=1, vel=False, alpha=0.8)
        
        if normalise_spectrum == True:
            ax.set_ylabel(r'Normalised Flux',fontsize=16,family='serif')
        else:
            ax.set_ylabel(r'Flux',fontsize=16,family='serif')

        #If user wants spectral lines overlaid
        if spectral_lines == True:
            if velocity_space == True:
                print("Can't plot spectral lines in velocity space (yet)")
            else:
                #Add emission markers
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()
                top_offset = np.subtract(ylim[1], ylim[0])*0.1
                top = ylim[1] - top_offset
                side_offset = np.subtract(xlim[1], xlim[0])*0.03

                ax.axvline(x=LINE_H_I[0],  color='black', linestyle='--', alpha=0.8)
                ax.text(LINE_H_I[0] + side_offset, top, r'H$\alpha$')
                ax.axvline(x=LINE_N_II[0], color='black', linestyle='--', alpha=0.8)
                ax.text(LINE_N_II[0] + side_offset, top, 'N II')
                ax.axvline(x=LINE_N_II[1], color='black', linestyle='--', alpha=0.8)
                ax.text(LINE_N_II[1] + side_offset, top, 'N II')
        # Add legend
            # data = mlines.Line2D([],[],color='black',marker='.', linestyle='none', label='Data')
            # fit = mlines.Line2D([],[],color='black', linestyle='-', label='Fit')
            # ax.legend(handles=[data, fit],loc=1)



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
