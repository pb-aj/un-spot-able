#general imports
import os
import sys
import starry2 as starry

# Directory structure
maindir    = os.path.dirname(os.path.realpath(__file__))
libdir     = os.path.join(maindir, 'lib')

# Lib imports
sys.path.append(libdir)
from lib import plots

# py imports
import create_eigens


starry.config.quiet = True

# Starry seems to have a lot of recursion
sys.setrecursionlimit(10000)

def create_emaps(star, eigeny, emaps_path):

    print("Plotting eigenmaps")
    eigenmaps = plots.emaps(star, eigeny, emaps_path, proj="rect")
    print("----------------------------------------------------------------------------")

    return eigenmaps

def create_eflux(star, eigeny, emaps_path):

    print("Calling create_eflux & plotting light curve")
    plots.eflux(star, eigeny, emaps_path, transparent=False)
    print("----------------------------------------------------------------------------")









if __name__ == "__main__":
    """
    """
    # print(sys.argv) #Uncomment if you want to see command line arguments 
    if len(sys.argv) < 2:
        print("\nERROR: Call structure is python create_emaps.py <configuration file>.")
        sys.exit()
    else:
        cfile = sys.argv[1]

    print("\nCalling create_eigens:")
    print("----------------------------------------------------------------------------")
    eigeny, evalues, evectors, ecurves, lcs, star, cfg, fit = \
        create_eigens.create_eigens(cfile)
    
    print("Setting current directory")
    subdir = cfg.folder
    outdir = os.path.join(cfg.outdir, subdir)
    emaps_path = os.path.join(outdir,"plots")

    if not os.path.isdir(emaps_path):
        os.mkdir(emaps_path)

    eigenmaps = create_emaps(star, eigeny, emaps_path)

    create_eflux(star, eigeny, emaps_path)

    print("done")

