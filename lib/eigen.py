import numpy as np
import pca

def mkcurves(star, nt, lmax, y00, ncurves=None, method='pca', negative = False, 
             remove_y00 = False, all_curves = False, factor_bool = True, start_l=1):
    """
    Generates light curves from a star+planet system at times t,
    for positive and negative spherical harmonics with l up to lmax.

    Arguments
    ---------
    star: object
        A starry star object, initialized with a cfg file

    t: 1D array
        Array of times at which to calculate eigencurves

    lmax: integer
        Maximum l to use in spherical harmonic maps

    y00: 1D array
        Light curve of a normalized, uniform map

    ncurves: integer
        Number of light curves to generate.  If left as None, it will equal nharm.
        nharm is ((lmax + 1)**2 - 1) by default but will be 
        ((lmax + 1)**2 - 1) * 2 if ngeative is True

    method: string
        Method to use for pca of light curve, by default it is pca.
        The other option is TSVD and is optimal for exoplanets

    negative: bool
        Designates if negative light curves should be generated.
        If True, negative curves will be made, otherwise only positive will be generated
        See factor_bool to learn about making only negative curves
        *NOTE* Need this to be True for TSVD method!

    remove_y00: bool
        Determines if the uniform map should be subtracted off.
        If True, y00 will be removed from each light curve.
        Otherwise there will be no subtraction.

    all_curves: bool
        When True, this will create eigenmaps for every curve.  
        *NOTE* This is only usable with method="pca" and negative=True
        When False, then only the inputted number of curves will be plotted.
    
    factor_bool: bool
        When True, positive maps will be used.  When False, negative maps are used.
        *NOTE* This is only possible when negative is False.
        

    Returns
    -------
    eigeny: 2D array
        nharm x ny array of y coefficients for each harmonic. nharm is
        the number of harmonics, including positive and negative versions
        and excluding Y00. That is, 2 * ((lmax + 1)**2 - 1). ny is the
        number of y coefficients to describe a harmonic with degree lmax.
        That is, (lmax + 1)**2.

    evalues: 1D array
        nharm length array of eigenvalues

    evectors: 2D array
        nharm x nt array of normalized (unit) eigenvectors

    proj: 2D array ("ecurves")
        nharm x nt array of the data projected in the new space (the PCA
        "eigencurves"). The imaginary part is discarded, if nonzero.

    lcs: 2D array
        ncurves x nt array of the light curves that are passed into pca
        to generate the other return values.
        *NOTE* ncurves may be 2xnharm or nharm depending on the value of negative 

    Extra Notes
    -------
    The following lists out ways to perform various pca calculations based on different input params.
    Generally, it is assumed that star, t, lmax, y00, & ncurves are changed by the user independent of these examples.

    Single Version PCA:
        Positive Changes - default parameters
        Negative Changes - Change factor_bool=False

    Exoplanet TSVD:
        Changes - Negative = True; remove_y00=True
    
    Standard PCA:
        Changes - Negative = True; remove_y00= True or False
    
    Complete PCA:
        Changes - Negative = True; remove_y00= True or False; all_curves = True

    File taken from Ryan Challener's github, 2024-10. Credit for
    Modifications made by A.J. deVaux.
    """    

    thet = np.linspace(0, 360, nt)

    if negative:
        # Create harmonic maps of the planet, excluding Y00
        # (lmax**2 maps, plus a negative version for all but Y00)
        nharm = ((lmax + 1)**2 - start_l) * 2
        lcs = np.zeros((nharm, nt))
        ind = 0
        for i, l in enumerate(range(1, lmax + 1)):
            for j, m in enumerate(range(-l, l + 1)):  
                star.map[l, m] =  1.0 #set the map to be positive
                lcs[ind] = star.map.flux(theta=thet).eval()
        

                star.map[l, m] = -1.0 #set the map to be negative
                lcs[ind+1] = star.map.flux(theta=thet).eval()


                star.map[l, m] = 0.0 #reset map to initial state
                ind += 2
       
    else:

        # Check for if only positive or negative maps should be made
        if factor_bool:
            factor = 1.0
        else:
            factor = -1.0
    

        # Create harmonic maps of the planet, excluding Y00
        # (lmax**2 maps - 1, no negative/positive maps depending on factor_bool)
        nharm = ((lmax + 1)**2 - start_l) 
        lcs = np.zeros((nharm, nt))
        ind = 0
        for i, l in enumerate(range(1, lmax + 1)):
            for j, m in enumerate(range(-l, l + 1)):  
                star.map[l, m] =  factor #set the map based on factor
                lcs[ind] = star.map.flux(theta=thet).eval()

                star.map[l, m] = 0.0 #reset map to initial state
                ind += 1

    # Subtact uniform map contribution if remove_y00 = True 
    # (starry includes this in all light curves)
    # (subtraction more useful in exoplanet mapping)
    if remove_y00:
        lcs -= y00
            
    # Run PCA to determine orthogonal light curves
    if ncurves is None:
        ncurves = nharm
    
    evalues, evectors, proj = pca.pca(lcs, method=method, ncomp=ncurves)

    # Discard imaginary part of eigencurves to appease numpy
    proj = np.real(proj)
    evalues = np.real(evalues)
    evectors = np.real(evectors)

    #Check negative value to determine how many curves exist
    if negative:

        #Double the number of curves depending on input parameters
        if method == 'pca' and all_curves:
            ncurves = ncurves * 2

        # Convert orthogonal light curves into maps
        eigeny = np.zeros((ncurves, (lmax + 1)**2))
        eigeny[:,0] = 1.0 # Y00 = 1 for all maps

        for j in range(ncurves):
            yi  = 1
            shi = 0
            for l in range(1, lmax + 1):
                for m in range(-l, l + 1):
                    # (ok because evectors has only been sorted along
                    #  one dimension)
                    eigeny[j,yi] = evectors.T[j,shi] - evectors.T[j,shi+1]
                    yi  += 1
                    shi += 2

    else:
        # Convert orthogonal light curves into maps
        eigeny = np.zeros((ncurves, (lmax + 1)**2))
        eigeny[:,0] = 1.0 # Y00 = 1 for all maps

        for j in range(ncurves):
            yi  = 1
            shi = 0
            for l in range(1, lmax + 1):
                for m in range(-l, l + 1):
                    # (ok because evectors has only been sorted along
                    #  one dimension)
                    eigeny[j,yi] = evectors.T[j,shi] #does not subtract negative part due to lack of curves
                    yi  += 1
                    shi += 1

    return eigeny, evalues, evectors, proj, lcs

"""
Function added by A.J. deVaux to pull out only the eigenmaps (no plotting)
"""

def mkmaps(star, eigeny, proj="rect"):

    ncurves, ny = eigeny.shape
    eigenmaps = []

    lmax = np.int(ny**0.5 - 1)

    for j in range(ncurves):
        star.map[1:,:] = 0
        
        yi = 1
        for l in range(1, lmax + 1):
            for m in range(-l, l + 1):
                star.map[l, m] = eigeny[j, yi]
                yi += 1
    
        eigenmaps.append(star.map.render(theta=0, projection=proj).eval())

    return np.array(eigenmaps)

            
