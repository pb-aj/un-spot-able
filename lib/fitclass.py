import os
import sys
import numpy as np
import pickle
import configparser as cp
import configclass as cc

# import utils

class Fit:
    """
    A class to hold attributes and methods related to fitting a model
    or set of models to data.
    """
    def read_config(self, cfile):
        """
        Read a configuration file and set up attributes accordingly.

        Note that self.cfg is a Configuration instance, and self.cfg.cfg
        is a raw ConfigParser instance. The ConfigParser instance should
        be parsed into attributes of the Configuration() instance for
        simpler access within other routines that use the Fit class.
        """
        config = cp.ConfigParser()
        config.read(cfile)
        self.cfg = cc.Configuration()

        self.cfg.cfile = cfile
        self.cfg.cfg   = config

        # General options
        self.cfg.outdir     = self.cfg.cfg.get('General', 'outdir')
        self.cfg.folder     = self.cfg.cfg.get('General', 'folder')

        #PCA options
        self.cfg.method     = self.cfg.cfg.get("PCA-Settings",'method')
        self.cfg.negative     = self.cfg.cfg.getboolean("PCA-Settings",'negative')
        self.cfg.remove_y00     = self.cfg.cfg.getboolean("PCA-Settings",'remove_y00')
        self.cfg.all_curves     = self.cfg.cfg.getboolean("PCA-Settings",'all_curves')
        self.cfg.factor_bool     = self.cfg.cfg.getboolean("PCA-Settings",'factor_bool')
        self.cfg.start_l      = self.cfg.cfg.getint("PCA-Settings",'start_l')
       
        # Gen options - Not needed
        # self.cfg.twod.timefile = self.cfg.cfg.get('Gen', 'timefile')
        # self.cfg.twod.fluxfile = self.cfg.cfg.get('Gen', 'fluxfile')
        # self.cfg.twod.ncurves = self.cfg.cfg.getint('Gen', 'ncurves')

        self.cfg.twod.lmax = self.cfg.cfg.getint('Gen', 'lmax')
        self.cfg.twod.nlcs = self.cfg.cfg.getint('Gen', 'nlcs')
        self.cfg.twod.nsamples = self.cfg.cfg.getint('Gen', 'nsamples')

        self.cfg.twod.udeg = [float(item.strip()) for item in self.cfg.cfg.get('Limb-Darkening', 'udeg').split(",")]

              
        # self.cfg.twod.posflux = self.cfg.cfg.getboolean('Gen', 'posflux')
        
        # self.cfg.twod.nlat    = self.cfg.cfg.getint('Gen', 'nlat')
        # self.cfg.twod.nlon    = self.cfg.cfg.getint('Gen', 'nlon')

        
        # self.cfg.twod.leastsq = self.cfg.cfg.get('Gen', 'leastsq')
        # if (self.cfg.twod.leastsq == 'None' or
        #     self.cfg.twod.leastsq == 'False'):
        #     self.cfg.twod.leastsq = None

        # if self.cfg.cfg.has_option('Gen', 'fgamma'):
        #     self.cfg.twod.fgamma = self.cfg.cfg.getfloat('Gen', 'fgamma')
        # else:
        #     self.cfg.twod.fgamma = 1.0

       
        # Star options
        self.cfg.star.m    = self.cfg.cfg.getfloat('Star', 'm')
        self.cfg.star.r    = self.cfg.cfg.getfloat('Star', 'r')
        self.cfg.star.prot = self.cfg.cfg.getfloat('Star', 'prot')
        # self.cfg.star.t    = self.cfg.cfg.getfloat('Star', 't')
        # self.cfg.star.d    = self.cfg.cfg.getfloat('Star', 'd')
        # self.cfg.star.z    = self.cfg.cfg.getfloat('Star', 'z')
        self.cfg.star.inc = self.cfg.cfg.getfloat('Star', 'inc')
        self.cfg.star.veq = self.cfg.cfg.getfloat('Star', 'veq')
        self.cfg.star.alpha = self.cfg.cfg.getfloat('Star', 'alpha')


    # def read_data(self):
        # self.t    = np.loadtxt(self.cfg.twod.timefile, ndmin=1)
        # self.flux = np.loadtxt(self.cfg.twod.fluxfile, ndmin=2).T
        # self.ferr = np.loadtxt(self.cfg.twod.ferrfile, ndmin=2).T

        # if len(self.t) != self.flux.shape[1]:
        #     print("WARNING: Number of times does not match the size " +
        #           "of the flux array.")
        #     sys.exit()

        # if len(self.t) != self.ferr.shape[1]:
        #     print("WARNING: Number of times does not match the size " +
        #           "of the ferr array.")
        #     sys.exit()

    def read_filters(self):
        self.filtwl, self.filtwn, self.filttrans, self.wnmid, self.wlmid = \
            utils.readfilters(self.cfg.twod.filtfiles)           
            
    def save(self, outdir, fname=None):
        # Note: starry objects are not pickleable, so they
        # cannot be added to the Fit object as attributes. Possible
        # workaround by creating a custom Pickler?
        if type(fname) == type(None):
            fname = 'fit.pkl'

        with open(os.path.join(outdir, fname), 'wb') as f:
            pickle.dump(self, f)

class Map:
    '''
    A class to hold results from a fit to a single wavelength (a 2d map).
    '''
    pass

def load(outdir=None, filename=None):
    """
    Load a Fit object from file.
    
    Arguments
    ---------
    outdir: string
        Location of file to load. Default is an empty string (current
        directory)

    filename: string
        Name of the file to load. Default is 'fit.pkl'.

    Returns
    -------
    fit: Fit instance
        Fit object loaded from filename
    """
    if type(outdir) == type(None):
        outdir = ''
        
    if type(filename) == type(None):
        filename = 'fit.pkl'
        
    with open(os.path.join(outdir, filename), 'rb') as f:
        return pickle.load(f)
