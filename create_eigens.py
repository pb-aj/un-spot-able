"""
need to run 
rm /Users/a.j.devaux/.theano/compiledir_macOS-26.4-arm64-arm-64bit-arm-3.10.17-64/.lock 
in terminal to get theano to work if saying constant folding error
"""


#general imports
import os
import sys
import numpy as np
import starry2 as starry

# Directory structure
maindir    = os.path.dirname(os.path.realpath(__file__))
libdir     = os.path.join(maindir, 'lib')

# Lib imports
sys.path.append(libdir)
print(sys.path)
from lib import eigen
from lib import utils
from lib import fitclass    as fc


starry.config.quiet = True

# Starry seems to have a lot of recursion
sys.setrecursionlimit(10000)


import faulthandler
faulthandler.enable()

def create_eigens(cfile):
    print("\nSetting-Up:")
    print("----------------------------------------------------------------------------")

    fit = fc.Fit()

    print("\tReading the configuration file & data")
    fit.read_config(cfile)
    cfg = fit.cfg

    lmax = cfg.twod.lmax
    udeg = cfg.twod.udeg
    nsamples = cfg.twod.nsamples

    star = utils.initstar(fit, lmax, udeg=udeg)
    print()
    print("\tFinding rank of the spherical harmponic design matrix:")
    print("\t------------------------------------------------")
    A = star.map.design_matrix(theta=np.linspace(0,360,100)).eval()

    R = np.empty((nsamples, lmax))

    for k in range(nsamples):
        R[k] = [
            np.linalg.matrix_rank(A[:, : (l + 1) ** 2]) for l in range(1, lmax + 1)
        ]

    R = np.median(R, axis=0)


    #Display rank for each spherical harmonic degree
    print(f"\t\033[1mThe rank (# non-null maps) for lmax = {lmax} (ncurves = {int(A.shape[1])}) is {int(R[-1])}\033[0m") 
    print("\t------------------------------------------------")
    print()

    print("\tCreating uniform star map")
    star = utils.initstar(fit, 1, udeg=udeg)

    fit.sflux = star.map.flux(theta=cfg.twod.nlcs).eval()

    print("\tSetting up new directory")
    m = fc.Map()
    m.ncurves = int((cfg.twod.lmax + 1) ** 2 - 1)
    m.lmax    = cfg.twod.lmax

    if not os.path.isdir(cfg.outdir):
        os.mkdir(cfg.outdir)

    m.subdir = cfg.folder
    outdir = os.path.join(cfg.outdir, m.subdir)

    if not os.path.isdir(os.path.join(cfg.outdir, m.subdir)):
        os.mkdir(os.path.join(cfg.outdir, m.subdir))

    print("\tRecreating star from cfg file")
    star = utils.initstar(fit, m.lmax, udeg=udeg)

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
        print(f"\tLoading in previously stored eigen results from:\n\t\033[34m{eigen_path}\033[0m")
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
        eigen.mkcurves(star, cfg.twod.nlcs, m.lmax, fit.sflux, ncurves=m.ncurves, method=cfg.method, \
                 negative = cfg.negative, remove_y00 = cfg.remove_y00, \
                 all_curves = cfg.all_curves, factor_bool = cfg.factor_bool, start_l=cfg.start_l)
        
        np.savetxt(f"{eigen_path}/eigeny.txt", eigeny)
        np.savetxt(f"{eigen_path}/evalues.txt", evalues)
        np.savetxt(f"{eigen_path}/evectors.txt", evectors)
        np.savetxt(f"{eigen_path}/ecurves.txt", ecurves)
        np.savetxt(f"{eigen_path}/lcs.txt",lcs)


    if not star.map.limbdark_is_physical().eval():
        print("----------------------------------------------------------------------------")
        print("WARNING: Limb Darkening is not physical!")


    print("----------------------------------------------------------------------------")
    print()
    print("Finding rank of the ecurve design matrix:")
    print("----------------------------------------------------------------------------")
    ecurve_A = ecurves.T

    ecurve_R = np.empty((nsamples, lmax))

    for k in range(nsamples):
        
        ecurve_R[k] = [
            np.linalg.matrix_rank(ecurve_A[:, : (l + 1) ** 2]) for l in range(1, lmax + 1)
        ]

    ecurve_R = np.median(ecurve_R, axis=0) 
    #Display rank for each spherical harmonic degree of the ecurve design matrix
    print(f"\t\033[1mThe rank (# non-null maps) for lmax = {lmax} (ncurves = {int(ecurve_A.shape[1])}) is {int(ecurve_R[-1])}\033[0m")
    print()
    print("\tNote, the ecurve design matrix does not include the uniform map\n\tand should be one less than the spherical harmonic result.")  
    print("----------------------------------------------------------------------------")

    return eigeny, evalues, evectors, ecurves, lcs, star, cfg, fit
    

    




if __name__ == "__main__":
    """
    """
    # print(sys.argv) #Uncomment if you want to see command line arguments 
    if len(sys.argv) < 2:
        print("ERROR: Call structure is python create_eigens.py <configuration file>.")
        sys.exit()
    else:
        cfile = sys.argv[1]

    create_eigens(cfile)
    
    print("done")

