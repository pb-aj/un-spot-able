"""
File to create eigens, check their rank, and save them for a stellar system 
[insert citation for my paper and also github here]


If theano shuts down and says 'constant folding error', may need to run line akin to:
rm /Users/[insert your username]/.theano/compiledir_macOS-26.4-arm64-arm-64bit-arm-3.10.17-64/.lock 
in terminal to get theano to work.  If this does not work, the error message should have 
the exact directory to insert after 'rm.'  Otherwise, post on the github.
"""


# General imports
import os
import sys
import numpy as np
import starry2 as starry
import faulthandler
faulthandler.enable()

# Directory structure
maindir    = os.path.dirname(os.path.realpath(__file__))
libdir     = os.path.join(maindir, 'lib')

# Lib imports
sys.path.append(libdir)
from lib import eigen
from lib import utils
from lib import fitclass as fc

# Set up starry configuration
starry.config.quiet = True
starry.config.lazy = True
sys.setrecursionlimit(10000) # starry seems to have a lot of recursion


def create_eigens(cfile):
    """
   [description]

    Arguments
    ---------
    cfile: string
        name of configuration file

    Returns
    -------
    eigeny: 2D array
        (ncurves, ny) array of y coefficients for each harmonic. 
        ncurves is the number of harmonics, including only positive versions and excluding Y00: (lmax + 1)**2 - 1
        ny is the number of y coefficients to describe a harmonic with degree lmax: (lmax + 1)**2 (includes y00)
    evalues: 1D array
        (ncurves) array of eigenvalues sorted by variance
    evectors: 2D array
        (ncurves, nlcs) array of normalized (unit) eigenvectors
    ecurves: 2D array ("proj")
        (ncurves, nt) array of the data projected in the new space (the PCA "eigencurves") 
        Note, the imaginary part is discarded
    lcs: 2D array
        (ncurves, nt) array of the light curves that are passed into pca to generate the other return values
    star: object
        A starry star object, initialized with a cfg file
    fit: fit class (defined in lib/fitclass.py)
        stored fit values read from cfg
    """    

    print("\nSetting-Up:")
    print("----------------------------------------------------------------------------")

    fit = fc.Fit()

    print("\tReading the configuration file & data")
    fit.read_config(cfile)
    cfg = fit.cfg

    lmax = cfg.twod.lmax
    udeg = cfg.twod.udeg

    star = utils.initstar(fit, lmax, udeg=udeg)
    print()
    print("\tFinding rank of the spherical harmponic design matrix:")
    print("\t------------------------------------------------")

    A = star.map.design_matrix(theta=np.linspace(0,360,100)).eval()
    R = np.empty((1, lmax))

    R = [
        np.linalg.matrix_rank(A[:, : (l + 1) ** 2]) for l in range(1, lmax + 1)
    ]

    #Display rank for each spherical harmonic degree
    print(f"\t\033[1mThe rank (# non-null maps) for lmax = {lmax} (ncurves = {int(A.shape[1])}) is {int(R[-1])}\033[0m") 
    print("\t------------------------------------------------")
    print()

    print("\tSetting up new directory")
    lmax = cfg.twod.lmax
    ncurves = int((lmax + 1) ** 2 - 1)
    nlcs = cfg.twod.nlcs

    if not os.path.isdir(cfg.outdir):
        os.mkdir(cfg.outdir)

    subdir = cfg.folder
    outdir = os.path.join(cfg.outdir, subdir)

    if not os.path.isdir(os.path.join(cfg.outdir, subdir)):
        os.mkdir(os.path.join(cfg.outdir, subdir))

    print()
    print("\tCreating star with parameters from cfg file")
    star = utils.initstar(fit, lmax, udeg=udeg)

    if not star.map.limbdark_is_physical().eval():
        print("----------------------------------------------------------------------------")
        print("\033[31mWARNING:\033[0m Limb Darkening is not physical!")

    eigen_path = os.path.join(outdir,"stored-eigens")
    if os.path.exists(eigen_path):
        print("----------------------------------------------------------------------------")
        print()
        use_stored_pca = input("Would you like to use the stored eigen results? (y/n) ")
    else:
        print("----------------------------------------------------------------------------")
        print()
        print("No stored eigen results found, creating new ones:")
        os.mkdir(eigen_path)
        use_stored_pca = "n"

    if use_stored_pca.lower().strip() == "y":
        print("----------------------------------------------------------------------------")
        print(f"\tReading previously stored eigen results from:\n\t\033[34m{eigen_path}\033[0m")
        try:
            eigeny = np.loadtxt(f"{eigen_path}/eigeny.txt")
            evalues = np.loadtxt(f"{eigen_path}/evalues.txt")
            evectors = np.loadtxt(f"{eigen_path}/evectors.txt")
            ecurves = np.loadtxt(f"{eigen_path}/ecurves.txt")
            lcs = np.loadtxt(f"{eigen_path}/lcs.txt")
        except:
            print()
            print(f"\t\033[31m\033[1mStored eigen results are invalid, calculating new ones!\033[0m")
            use_stored_pca = "n"

    if use_stored_pca.lower().strip() != "y":
        print("----------------------------------------------------------------------------")
        print(f"\tCalculating new eigen results and storing them in:\n\t\033[34m{eigen_path}\033[0m")
        eigeny, evalues, evectors, ecurves, lcs = \
        eigen.mkcurves(star, nlcs, lmax, ncurves, cfg.twod.use_y00)
        
        np.savetxt(f"{eigen_path}/eigeny.txt", eigeny)
        np.savetxt(f"{eigen_path}/evalues.txt", evalues)
        np.savetxt(f"{eigen_path}/evectors.txt", evectors)
        np.savetxt(f"{eigen_path}/ecurves.txt", ecurves)
        np.savetxt(f"{eigen_path}/lcs.txt",lcs)


    print("----------------------------------------------------------------------------")
    print()
    print("Finding rank of the ecurve design matrix:")
    print("----------------------------------------------------------------------------")
    ecurve_A = ecurves.T

    ecurve_R = np.empty((1, lmax))
        
    ecurve_R = [
        np.linalg.matrix_rank(ecurve_A[:, : (l + 1) ** 2]) for l in range(1, lmax + 1)
    ]

    #Display rank for each spherical harmonic degree of the ecurve design matrix
    print(f"\t\033[1mThe rank (# non-null maps) for lmax = {lmax} (ncurves = {int(ecurve_A.shape[1])}) is {int(ecurve_R[-1])}\033[0m")
    print()
    print("\tNote, the ecurve design matrix does not include the uniform map\n\tand thus the rank should one less than the spherical harmonic result.")  
    print("----------------------------------------------------------------------------")
    return eigeny, evalues, evectors, ecurves, lcs, star, fit
    

    




if __name__ == "__main__":
    """
    """
    # print(sys.argv) #Uncomment if you want to see command line arguments 
    if len(sys.argv) < 2:
        print()
        print("----------------------------------------------------------------------------")
        print('\033[31mERROR:\033[0m' + ' Call structure is "\033[34mpython create_eigens.py <configuration file>\033[0m"')
        print("----------------------------------------------------------------------------")
        sys.exit()
    else:
        cfile = sys.argv[1]

    create_eigens(cfile)
    
    print("done")

