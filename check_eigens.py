#general imports
import os
import sys
import starry2 as starry
import numpy as np
import matplotlib.pyplot as plt


# Directory structure
maindir    = os.path.dirname(os.path.realpath(__file__))
libdir     = os.path.join(maindir, 'lib')

# Lib imports
sys.path.append(libdir)
print(sys.path)
from lib import constants
from lib import pca
from lib import eigen
from lib import plots
from lib import utils
from lib import constants   as c
from lib import fitclass    as fc

# py imports
import create_eigens
import create_emaps

#Setting up colored print statements
class style():
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'


starry.config.quiet = True

# Starry seems to have a lot of recursion
sys.setrecursionlimit(10000)

#Fix GUI backend that is changed by mc3
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def check_ortho(ecurves):

    print("\nChecking if ecurves are orthogonal:")
    print("----------------------------------------------------------------------------")

    zero_error = 1e-8

    non_ortho = {}

    dot_results = {}

    for i in range(ecurves.shape[0]):
        for j in range(ecurves.shape[0]):
            if i != j:
                dot_result = abs(np.dot(ecurves[i],ecurves[j]))
                if  dot_result > zero_error:
                    non_ortho[i] = non_ortho.get(i,[]) + [j]

                    if i < j:
                        angle = np.arccos((dot_result / np.linalg.norm(ecurves[i]) / np.linalg.norm(ecurves[j]))) * 180 / np.pi
                        dot_results[f"({i},{j})"] = [dot_result, angle]


    if len(non_ortho) != 0:
        print(f"\tThe following ecurves are {style.RED}not orthogonal{style.RESET}:")
        print(f"\t{non_ortho}")
        print()
    else:
        print(f"\tAll curves are {style.GREEN}orthogonal{style.RESET}")

        return non_ortho, dot_results
    
    see_curves = input("\tWould you like to see the error? (y/n) ")

    if see_curves.lower().strip() == "y":
        print("\tThe results of the non-orthogonal dot products are:")

        for x, y in dot_results.items():
            print(f"\t{x} : {y[0]:.2E} or {y[1]} degrees")
        
        return non_ortho, dot_results
    else:
        return non_ortho, dot_results



def check_erank(evalues,outdir):

    print("\nChecking order of evalue rank:")
    print("----------------------------------------------------------------------------")

    print("\tGenerating plot of evalues")

    plt.plot(evalues,"go-")
    plt.xlabel("Evalue Rank")
    plt.title("Distribution of Evalues")

    checks_path = os.path.join(outdir,"checks")
    if not os.path.isdir(checks_path):
        os.mkdir(checks_path)

    plt.savefig(f"{checks_path}/evalue_rank.png", bbox_inches="tight",dpi=300)


    print("\tChecking rank numerically")
    for i in range(len(evalues)-1):
        current = evalues[i]
        plus_one = evalues[i+1]

        
        if current - plus_one < -1e-15:
            print(f"\tThe evalues are {style.RED}out of order{style.RESET}; {i} is smaller than {i+1}!")
            return None
        
    print(f"\tThe evalue are in the {style.GREEN}correct order{style.RESET}.")

    pass


def check_consistency(cfile, outdir):

    print("\nChecking if eigens are unchanged on each run:")
    print("----------------------------------------------------------------------------")

    fit = fc.Fit()

    fit.read_config(cfile)
    cfg = fit.cfg

    udeg = cfg.twod.udeg

    star = utils.initstar(fit, 1, udeg=udeg)

    fit.sflux = star.map.flux(theta=cfg.twod.nlcs).eval()

    m = fc.Map()
    m.ncurves = int((cfg.twod.lmax + 1) ** 2 - 1)
    m.lmax    = cfg.twod.lmax

    star = utils.initstar(fit, m.lmax, udeg=udeg)

    eigeny, evalues, evectors, ecurves, lcs = \
        eigen.mkcurves(star, cfg.twod.nlcs, m.lmax, fit.sflux, ncurves=m.ncurves, method=cfg.method, \
                 negative = cfg.negative, remove_y00 = cfg.remove_y00, \
                 all_curves = cfg.all_curves, factor_bool = cfg.factor_bool, start_l=cfg.start_l)
    
    eigeny2, evalues2, evectors2, ecurves2, lcs2 = \
        eigen.mkcurves(star, cfg.twod.nlcs, m.lmax, fit.sflux, ncurves=m.ncurves, method=cfg.method, \
                 negative = cfg.negative, remove_y00 = cfg.remove_y00, \
                 all_curves = cfg.all_curves, factor_bool = cfg.factor_bool, start_l=cfg.start_l)
    
    checks_path = os.path.join(outdir,"checks")
    if not os.path.isdir(checks_path):
        os.mkdir(checks_path)
    
    eigenmaps = plots.emaps(star, eigeny, f"{checks_path}", proj="rect", plot_name='emaps-check1.png')

    eigenmaps2 = plots.emaps(star, eigeny2, f"{checks_path}", proj="rect", plot_name='emaps-check2.png')
    

    method = cfg.method
    any_incon = False

    print("The results are:")
    if np.array_equal(lcs,lcs2):
        print(f"\tLCS - {style.GREEN}Consistent{style.RESET}.")
        lcs_const = True
    else:
        print(f"\tLCS - {style.RED}Inconsistent{style.RESET}.")
        lcs_const = False
        any_incon = True

    if np.array_equal(eigeny,eigeny2):
        print(f"\tEigeny - {style.GREEN}Consistent{style.RESET}.")
        eigeny_const = True
    else:
        print(f"\tEigeny - {style.RED}Inconsistent{style.RESET}.")
        eigeny_const = False
        any_incon = True

    if np.array_equal(evalues,evalues2):
        print(f"\tEvalues - {style.GREEN}Consistent{style.RESET}.")
        evalues_const = True
    else:
        print(f"\tEvalues - {style.RED}Inconsistent{style.RESET}.")
        evalues_const = False
        any_incon = True

    if np.array_equal(evectors.T,evectors2.T):
        print(f"\tEvectors - {style.GREEN}Consistent{style.RESET}.")
        evectors_const = True
    else:
        print(f"\tEvectors - {style.RED}Inconsistent{style.RESET}.")
        evectors_const = False
        any_incon = True

    if np.array_equal(ecurves,ecurves2):
        print(f"\tEcurves - {style.GREEN}Consistent{style.RESET}.")
        ecurves_const = True
    else:
        print(f"\tEcurves - {style.RED}Inconsistent{style.RESET}.")
        ecurves_const = False
        any_incon = True

    if np.array_equal(eigenmaps,eigenmaps2):
        print(f"\tEigenmaps - {style.GREEN}Consistent{style.RESET}.")
        emaps_const = True
    else:
        print(f"\tEigenmaps - {style.RED}Inconsistent{style.RESET}.")
        emaps_const = False
        any_incon = True

    print("Basic Checks Completed!  The eigenmap plots are stored in 'checks' folder.")

    """Need to fix sorted using np.lexsort not np.sort"""

    # if any_incon:
    #     print("\nChecking if it is a sorting issue:")
    #     print("----------------------------------------------------------------------------")
        
    #     print("The results after sorting are:")
    #     if not lcs_const:
    #         if np.array_equal(np.sort(lcs),np.sort(lcs2)):
    #             print(f"\tLCS - {style.GREEN}Consistent{style.RESET}.")
    #         else:
    #             print(f"\tLCS - {style.RED}Inconsistent{style.RESET}.")

    #     if not eigeny_const:
    #         if np.array_equal(np.sort(eigeny),np.sort(eigeny2)):
    #             print(f"\tEigeny - {style.GREEN}Consistent{style.RESET}.")
    #         else:
    #             print(f"\tEigeny - {style.RED}Inconsistent{style.RESET}.")

    #     if not evalues_const:
    #         if np.array_equal(np.sort(evalues),np.sort(evalues2)):
    #             print(f"\tEvalues - {style.GREEN}Consistent{style.RESET}.")
    #         else:
    #             print(f"\tEvalues - {style.RED}Inconsistent{style.RESET}.")

    #     if not evectors_const:
    #         if np.array_equal(np.sort(evectors.T),np.sort(evectors2.T)):
    #             print(f"\tEvectors - {style.GREEN}Consistent{style.RESET}.")
    #         else:
    #             print(f"\tEvectors - {style.RED}Inconsistent{style.RESET}.")

    #     if not ecurves_const:
    #         if np.array_equal(np.sort(ecurves),np.sort(ecurves2)):
    #             print(f"\tEcurves - {style.GREEN}Consistent{style.RESET}.")
    #         else:
    #             print(f"\tEcurves - {style.RED}Inconsistent{style.RESET}.")

    #     if not emaps_const:
    #         if np.array_equal(np.sort(eigenmaps),np.sort(eigenmaps2)):
    #             print(f"\tEigenmaps - {style.GREEN}Consistent{style.RESET}.")
    #         else:
    #             print(f"\tEigenmaps - {style.RED}Inconsistent{style.RESET}.")




def check_repeats(eigeny, evalues, evectors, ecurves):

    print("\nChecking if any eigen values are repeated:")
    print("----------------------------------------------------------------------------")

    eigeny_repeats = {}

    for i in range(eigeny.shape[0]):
        for j in range(eigeny.shape[0]):
            if i < j:
                if np.array_equal(eigeny[i],eigeny[j]):
                    eigeny_repeats[i] = eigeny_repeats.get(i,[]) + [j]

    evalue_repeats = {}

    for i in range(evalues.shape[0]):
        for j in range(evalues.shape[0]):
            if i < j:
                if np.array_equal(evalues[i],evalues[j]):
                    evalue_repeats[i] = evalue_repeats.get(i,[]) + [j]


    evector_repeats = {}

    for i in range(evectors.T.shape[0]):
        for j in range(evectors.T.shape[0]):
            if i < j:
                if np.array_equal(evectors.T[i],evectors.T[j]):
                    evector_repeats[i] = evector_repeats.get(i,[]) + [j]


    ecurves_repeats = {}

    for i in range(ecurves.shape[0]):
        for j in range(ecurves.shape[0]):
            if i < j:
                if np.array_equal(ecurves[i],ecurves[j]):
                    ecurves_repeats[i] = ecurves_repeats.get(i,[]) + [j]

    # emap_repeats = {}

    # for i in range(eigenmaps.shape[0]):
    #     for j in range(eigenmaps.shape[0]):
    #         if i < j:
    #             if np.array_equal(eigenmaps[i],eigenmaps[j]):
    #                 emap_repeats[i] = emap_repeats.get(i,[]) + [j]

    lcs_repeats = {}

    for i in range(lcs.shape[0]):
        for j in range(lcs.shape[0]):
            if i < j:
                if np.array_equal(lcs[i],lcs[j]):
                    lcs_repeats[i] = lcs_repeats.get(i,[]) + [j]

    print("The results are:")

    if lcs_repeats:
        print(f"\tLCS - {style.RED}Repeats{style.RESET} as follows:")
        for k,v in lcs_repeats.items():
            print(f"\t[{k}] - {v}")
    else:
        print(f"\tLCS - {style.GREEN}No Repeats{style.RESET}.")

    if eigeny_repeats:
        print(f"\tEigeny - {style.RED}Repeats{style.RESET} as follows:")
        for k,v in eigeny_repeats.items():
            print(f"\t[{k}] - {v}")
    else:
        print(f"\tEigeny - {style.GREEN}No Repeats{style.RESET}.")

    if evalue_repeats:
        print(f"\tEvalues - {style.RED}Repeats{style.RESET} as follows:")
        for k,v in evalue_repeats.items():
            print(f"\t[{k}] - {v}")
    else:
        print(f"\tEvalues - {style.GREEN}No Repeats{style.RESET}.")

    if evector_repeats:
        print(f"\tEvectors - {style.RED}Repeats{style.RESET} as follows:")
        for k,v in evector_repeats.items():
            print(f"\t[{k}] - {v}")
    else:
        print(f"\tEvectors- {style.GREEN}No Repeats{style.RESET}.")

    if ecurves_repeats:
        print(f"\tEcurves - {style.RED}Repeats{style.RESET} as follows:")
        for k,v in ecurves_repeats.items():
            print(f"\t[{k}] - {v}")
    else:
        print(f"\tEcurves - {style.GREEN}No Repeats{style.RESET}.")

    # if emap_repeats:
    #     print(f"\tEigenmaps - {style.RED}Repeats{style.RESET} as follows:")
    #     for k,v in emap_repeats.items():
    #         print(f"\t[{k}] - {v}")
    # else:
    #     print(f"\tEigenmaps - {style.GREEN}No Repeats{style.RESET}.")

    





def check_inc_change(eigeny, evalues, evectors, ecurves, lcs, star, cfile, outdir):

    print("\nChecking if inclination affects the eigen null space:")
    print("----------------------------------------------------------------------------")

    fit = fc.Fit()

    fit.read_config(cfile)
    cfg = fit.cfg
    fit.read_data()

    udeg = cfg.twod.udeg

    init_inc = cfg.star.inc
    random_inc = cfg.star.inc - 45

    if random_inc < 5:
        random_inc += 90


    cfg.star.inc = random_inc

    star2 = utils.initstar(fit, 1, udeg=udeg)

    fit.sflux = star2.map.flux(theta=cfg.twod.nlcs).eval()

    m = fc.Map()
    m.ncurves = int((cfg.twod.lmax + 1) ** 2 - 1)
    m.lmax    = cfg.twod.lmax

    start_index = int(4 * np.floor(m.lmax/2))

    star2 = utils.initstar(fit, m.lmax, udeg=udeg)

    eigeny2, evalues2, evectors2, ecurves2, lcs2 = \
        eigen.mkcurves(star2, cfg.twod.nlcs, m.lmax, fit.sflux, ncurves=m.ncurves, method=cfg.method, \
                 negative = cfg.negative, remove_y00 = cfg.remove_y00, \
                 all_curves = cfg.all_curves, factor_bool = cfg.factor_bool, start_l=cfg.start_l)
    
    checks_path = os.path.join(outdir,"checks")
    if not os.path.isdir(checks_path):
        os.mkdir(checks_path)
    
    saved_checks_path = os.path.join(outdir,"checks/inclination_eigens")
    if not os.path.isdir(saved_checks_path):
        os.mkdir(saved_checks_path)
    
    np.savetxt(f"{saved_checks_path}/eigeny.txt", eigeny2)
    np.savetxt(f"{saved_checks_path}/evalues.txt", evalues2)
    np.savetxt(f"{saved_checks_path}/evectors.txt", evectors2)
    np.savetxt(f"{saved_checks_path}/ecurves.txt", ecurves2)
    np.savetxt(f"{saved_checks_path}/lcs.txt",lcs2)

    eigenmaps = plots.emaps(star, eigeny, f"{checks_path}", proj="rect", plot_name=f'inc{init_inc}-emaps.png')

    eigenmaps2 = plots.emaps(star2, eigeny2, f"{checks_path}", proj="rect", plot_name=f'inc{random_inc}-emaps.png')

    any_icon = True

    print("The results are:")
    if not np.array_equal(lcs,lcs2):
        print(f"\tLCS - {style.GREEN}Changes{style.RESET}.")
        lcs_const = False
        any_icon = False
    else:
        print(f"\tLCS - {style.RED}No Change{style.RESET}.")
        lcs_const = True

    if not np.array_equal(eigeny[start_index:],eigeny2[start_index:]):
        print(f"\tEigeny - {style.GREEN}Changes{style.RESET}.")
        eigeny_const = False
        any_icon = False
    else:
        print(f"\tEigeny - {style.RED}No Change{style.RESET}.")
        eigeny_const = True

    if not np.array_equal(evalues[start_index:],evalues2[start_index:]):
        print(f"\tEvalues - {style.GREEN}Changes{style.RESET}.")
        evalues_const = False
        any_icon = False
    else:
        print(f"\tEvalues - {style.RED}No Change{style.RESET}.")
        evalues_const = True

    if not np.array_equal(evectors.T[start_index:],evectors2.T[start_index:]):
        print(f"\tEvectors - {style.GREEN}Changes{style.RESET}.")
        evectors_const = False
        any_icon = False
    else:
        print(f"\tEvectors - {style.RED}No Change{style.RESET}.")
        evectors_const = True

    if not np.array_equal(ecurves[start_index:],ecurves2[start_index:]):
        print(f"\tEcurves - {style.GREEN}Changes{style.RESET}.")
        ecurves_const = False
        any_icon = False

    else:
        print(f"\tEcurves - {style.RED}No Change{style.RESET}.")
        ecurves_const = True

    # if not np.array_equal(eigenmaps[start_index:],eigenmaps2[start_index:]):
    #     print(f"\tEigenmaps - {style.GREEN}Changes{style.RESET}.")

    # else:
    #     print(f"\tEigenmaps - {style.RED}No Change{style.RESET}.")
    #     any_icon = False

    """Need to fix sorted using np.lexsort not np.sort"""

    # if not any_icon:
    #     print("\nChecking if it is a sorting issue:")
    #     print("----------------------------------------------------------------------------")
        
    #     print("The results after sorting are:")

    #     if not lcs_const:
    #         if not np.array_equal(np.sort(lcs),np.sort(lcs2)):
    #             print(f"\tLCS - {style.GREEN}Changes{style.RESET}.")
    #         else:
    #             print(f"\tLCS - {style.RED}No Change{style.RESET}.")

    #     if not eigeny_const:
    #         if not np.array_equal(np.sort(eigeny[start_index:]),np.sort(eigeny2[start_index:])):
    #             print(f"\tEigeny - {style.GREEN}Changes{style.RESET}.")
    #         else:
    #             print(f"\tEigeny - {style.RED}No Change{style.RESET}.")

    #     if not evalues_const:
    #         if not np.array_equal(np.sort(evalues[start_index:]),np.sort(evalues2[start_index:])):
    #             print(f"\tEvalues - {style.GREEN}Changes{style.RESET}.")
    #         else:
    #             print(f"\tEvalues - {style.RED}No Change{style.RESET}.")

    #     if not evectors_const:
    #         if not np.array_equal(np.sort(evectors.T[start_index:]),np.sort(evectors2.T[start_index:])):
    #             print(f"\tEvectors - {style.GREEN}Changes{style.RESET}.")
    #         else:
    #             print(f"\tEvectors - {style.RED}No Change{style.RESET}.")

    #     if not ecurves_const:
    #         if not np.array_equal(np.sort(ecurves[start_index:]),np.sort(ecurves2[start_index:])):
    #             print(f"\tEcurves - {style.GREEN}Changes{style.RESET}.")
    #         else:
    #             print(f"\tEcurves - {style.RED}No Change{style.RESET}.")

    print("Inclination Checks Completed!  The eigenmap plots are stored in 'checks' folder.")






if __name__ == "__main__":
    """
    """
    # print(sys.argv) #Uncomment if you want to see command line arguments 
    if len(sys.argv) < 2:
        print("\nERROR: Call structure is python check_eigens.py <configuration file>.")
        sys.exit()
    else:
        cfile = sys.argv[1]

    eigeny, evalues, evectors, ecurves, lcs, star, cfg, fit = \
        create_eigens.create_eigens(cfile)
    
    print("Setting current directory")
    subdir = cfg.folder
    outdir = os.path.join(cfg.outdir, subdir)

    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    print("----------------------------------------------------------------------------")

    ortho_input = input("Would you like to check the ecurve orthogonality? (y/n) ")

    if ortho_input.lower().strip() == "y":
        non_ortho, dot_results = check_ortho(ecurves)

    print("----------------------------------------------------------------------------")

    rank_input = input("Would you like to check the evalue rank? (y/n) ")

    if rank_input.lower().strip() == "y":
        check_erank(evalues, outdir)

    print("----------------------------------------------------------------------------")

    const_input = input("Would you like to check the consistentcy of the eigens? (y/n) ")

    if const_input.lower().strip() == "y":
        check_consistency(cfile,outdir)

    print("----------------------------------------------------------------------------")

    repeat_input = input("Would you like to check if any of the eigens have repeating values? (y/n) ")

    if repeat_input.lower().strip() == "y":
        check_repeats(eigeny, evalues, evectors, ecurves)

    print("----------------------------------------------------------------------------")

    inc_input = input("Would you like to check if changing the inclination changes the eigens? (y/n) ")

    if inc_input.lower().strip() == "y":
        check_inc_change(eigeny, evalues, evectors, ecurves, lcs, star, cfile, outdir)

    print("----------------------------------------------------------------------------")


    print("done")