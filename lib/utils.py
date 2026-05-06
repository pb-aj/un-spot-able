"""
File to set up stellar system and parameters in starry
Code sourced from: https://github.com/rychallener/theresa
Challener, R. C., & Rauscher, E. 2022, AJ, 163, 117, doi: 10.3847/1538-3881/ac4885

Modifications made by A.J. deVaux (https://github.com/pb-aj/un-spot-able)
"""

import numpy as np
import starry2 as starry
from spotable import spotable as se

# Whether or not to show print statements in code
dpm = True #change to False to make quiet

def initstar(fit, ydeg, udeg=[], include_rv=False):
    """
    Uses a fit object to build the respective starry objects. Useful
    because starry objects cannot be pickled. Returns a star object.

    Arguments
    ---------
    fit: fit class (defined in lib/fitclass.py)
        stored fit values read from cfg

    ydeg: integer
        maximum spherical harmonic degree of the star's surface
        Sets number of modes (map.y in starry) to (lmax + 1)**2

    udeg: 1D array (optional)
        limb darkening coefficient to use for star
        If empty array (default), no limb darkening is applied.  Goes up to quadratic (2 coefficients)

    include_rv: boolean (optional)
        Whether to include radial velocity map features in star.  Default to False
        If True, will use fit to set veq of star

    Returns
    -------
    star: object
        A starry star object, initialized with a cfg file

    """
    # Read in configuration file
    cfg = fit.cfg

    # Determine if limb darkening is being applied
    # need to do this as 0 is used for no limb darkening in cfg
    if len(udeg) == 1 and udeg[0] == 0:
        udeg = []

    # Set up star object through starry
    star = starry.Primary(starry.Map(ydeg=ydeg,udeg=len(udeg), rv=include_rv),
                          r   =cfg.star.r,
                          prot= cfg.star.prot,
                          theta0=180)

    star.map.inc = cfg.star.inc
    
    # Set limb darkening parameters if any
    for i in range(len(udeg)):
        star.map[i+1] = udeg[i]

    # Set equitorial velocity of star object
    if include_rv:
        try:
            veq_calc = ((star.r * 6.957e8) * np.pi * 2 / (star.prot * 86400)).eval()
        except:
            veq_calc = False

        if cfg.star.veq == 0:
            star.map.veq = float('%.2g' % veq_calc)
        else:
            star.map.veq = cfg.star.veq

            if not veq_calc * .95 <= cfg.star.veq <= veq_calc * 1.05:
                se("----------------------------------------------------------------------------",dp = dpm)
                se("\033[31mWARNING:\033[0m Equitorial Velocity is not physical!",dp = dpm)
                se("----------------------------------------------------------------------------",dp = dpm)

    return star

"""
[MAY REMOVE THIS]
"""
def initsystem(fit, ydeg):
    '''
    Uses a fit object to build the respective starry objects. Useful
    because starry objects cannot be pickled. Returns a tuple of
    (star, planet, system).
    '''
    
    cfg = fit.cfg

    star = starry.Primary(starry.Map(ydeg=1, amp=1),
                          m   =cfg.star.m,
                          r   =cfg.star.r,
                          prot=cfg.star.prot)

    planet = starry.kepler.Secondary(starry.Map(ydeg=ydeg),
                                     m    =cfg.planet.m,
                                     r    =cfg.planet.r,
                                     porb =cfg.planet.porb,
                                     prot =cfg.planet.prot,
                                     Omega=cfg.planet.Omega,
                                     ecc  =cfg.planet.ecc,
                                     w    =cfg.planet.w,
                                     t0   =cfg.planet.t0,
                                     inc  =cfg.planet.inc,
                                     theta0=180)

    system = starry.System(star, planet)

    return star, planet, system
