import numpy as np
import pca

"""
Add citation here:
File taken from Ryan Challener's github, 2024-10. Credit for
Modifications made by A.J. deVaux.
"""

def mkcurves(star, nt, lmax, ncurves, method='pca'):
    """
    Generates light curves for each spherical harmonic mode (excludin Y00) of passed star.
    Runs the light curve design matrix (lcs) through pca to generate eigencurve design matrix
    and eigenmaps with observable and null components separated

    Arguments
    ---------
    star: object
        A starry star object, initialized with a cfg file

    nt: integer
        Length of each light curve array

    lmax: integer
        Maximum spherical harmonic degree (l) to use

    ncurves: integer
        Number of light curves to generate.  
        If left as None, it will equal nharm which is (lmax + 1)**2 - 1

    method: string
        Method to use for pca of light curve, by default it is full_pca.
        The other options are not well tested in this code

    Returns
    -------
    eigeny: 2D array
        ncurves x ny array of y coefficients for each harmonic. 
        ncurves is the number of harmonics, including only positive versions and excluding Y00: lmax + 1)**2 - 1. 
        ny is the number of y coefficients to describe a harmonic with degree lmax: (lmax + 1)**2.

    evalues: 1D array
        ncurves length array of eigenvalues sorted by variance

    evectors: 2D array
        ncurves x nt array of normalized (unit) eigenvectors

    proj: 2D array ("ecurves")
        ncurves x nt array of the data projected in the new space (the PCA "eigencurves"). 
        The imaginary part is discarded.

    lcs: 2D array
        ncurves x nt array of the light curves that are passed into pca to generate the other return values.
    """    


    thet = np.linspace(0, 360, nt)
    nharm = ncurves
    lcs = np.zeros((nharm, nt))
    ind = 0

    # Loop through each harmonic map of the star, excluding Y00
    for i, l in enumerate(range(1, lmax + 1)):
        for j, m in enumerate(range(-l, l + 1)):  
            star.map[l, m] =  1 #set the map based on factor
            lcs[ind] = star.map.flux(theta=thet).eval()

            star.map[l, m] = 0.0 #reset map to initial state
            ind += 1

            
    # Run PCA to determine orthogonal light curves
    evalues, evectors, proj = pca.pca(lcs, method=method, ncomp=nharm)

    # Discard imaginary part of eigencurves to appease numpy
    proj = np.real(proj)
    evalues = np.real(evalues)
    evectors = np.real(evectors)

    # Convert orthogonal light curves into maps
    eigeny = np.zeros((ncurves, (lmax + 1)**2))
    eigeny[:,0] = 1.0 # Y00 = 1 for all maps

    for j in range(ncurves):
        yi  = 1
        shi = 0
        for l in range(1, lmax + 1):
            for m in range(-l, l + 1):
                # (ok because evectors has only been sorted along one dimension)
                eigeny[j,yi] = evectors.T[j,shi]
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

            
