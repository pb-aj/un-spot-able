#general imports
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import starry2 as starry

import matplotlib.ticker as ticker

import faulthandler
faulthandler.enable()


# Directory structure
maindir    = os.path.dirname(os.path.realpath(__file__))
libdir     = os.path.join(maindir, 'lib')

# Lib imports
sys.path.append(libdir)
print(sys.path)
from lib import utils

# py imports
import create_eigens

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

def custom_formatter(x, pos):
    return f"{round(x, 2):.2f}" # Rounds to nearest integer

def adjust_star(rv_star,eigeny_comp,lower_limit=.1, factor = None, iterator = 1):

    if factor:
        if type(factor) == int:
            rv_star.map[1:,:] = 0

            yi = 1
            for l in range(1, rv_star.map.ydeg + 1):
                for m in range(-l, l + 1):
                    rv_star.map[l, m] = eigeny_comp[yi] / factor
                    yi += 1
            return rv_star, factor
        else:
            print("Invalid Factor, instead scaling to default lower limit.")
            pass


    factor = 1

    min_val = rv_star.map.minimize()[-1].eval()
    while min_val < lower_limit:
            rv_star.map[1:,:] = 0

            factor += iterator

            yi = 1
            for l in range(1, rv_star.map.ydeg + 1):
                for m in range(-l, l + 1):
                    rv_star.map[l, m] = eigeny_comp[yi] / factor
                    yi += 1
            min_val = rv_star.map.minimize()[-1].eval()
            

    return rv_star, factor


def rv_flux_map_ani(rv_star,theta,fname=None):

    fig, axes = plt.subplots(nrows=1, ncols=2, squeeze=False,
                                    sharex=False, sharey=False, figsize=(12, 6))
        
    img1,image1,lonlines1,latlines1 = rv_star.map.show(theta=theta,rv=False, ax=axes[0,0],show_image=False,)
    rv_star.map.show(theta=theta,rv=True, ax=axes[0,1],show_image=True, 
                        extra_image=[img1,image1,lonlines1,latlines1],file=fname, dpi = 300)
    

def flux_rv_line(rv_star,theta,flux_name=None,rv_name=None,flux_only=False,rv_only=False):
    if not rv_only:
        # Plot the flux
        plt.figure(figsize=(12, 5))
        plt.plot(theta, rv_star.map.flux(theta=theta).eval())
        plt.xlabel("Angle of rotation [degrees]", fontsize=24)
        plt.ylabel("Flux [normalized]", fontsize=24)
        plt.tight_layout()
        if flux_name:
            plt.savefig(flux_name, dpi = 300)
        else:
            plt.show()
        plt.close()

    if not flux_only:
        # Plot the radial velocity
        plt.figure(figsize=(12, 5))
        plt.plot(theta, rv_star.map.rv(theta=theta).eval())
        plt.xlabel("Angle of rotation [degrees]", fontsize=24)
        plt.ylabel("Radial velocity [m/s]", fontsize=24)
        plt.tight_layout()
        if rv_name:
            plt.savefig(rv_name, dpi = 300)
        else:
            plt.show()
        plt.close()



def multi_phase_plot(rv_star,fname=None):

    nrows = 3
    ncols = 4

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols*2, squeeze=False,
                                sharex=False, sharey=False, figsize=(12, 6))
    
    degree = 0
    number_plots = nrows*ncols
    for j in range(number_plots):

        xloc = j %  ncols
        yloc = j // ncols
        ax = axes[yloc, xloc]
        ax2 = axes[yloc, xloc+ncols]


        # print(rv_star.map.rv(theta=i).eval(),rv_star.map.flux(theta=i).eval())

        rv_star.map.show(rv=False,theta=degree,ax=ax,figsize=(5,5))
        ax.set_title(f"{rv_star.map.flux(theta=degree).eval()[0]:.3f}")

        rv_star.map.show(rv=True,theta=degree,ax=ax2,figsize=(5,5))
        ax2.set_title(f"{rv_star.map.rv(theta=degree).eval()[0]:.3f}")

        degree += int(360/number_plots)
    
    fig.text(.22, 0.95, f"Flux Maps",fontsize="large")
    fig.text(.72, 0.95, f"RV Maps",fontsize="large")
    fig.suptitle(f"Plots range from 0 to 360 in {int(360/number_plots)} degree intervals.",y=0.02)
    fig.tight_layout()
    if fname:
        plt.savefig(fname,bbox_inches="tight",dpi=300)
    else:
        plt.show()
    plt.close()


def map_curve_ani(rv_star,theta,fname=None,cbar_pos="bottom",transparent=False):

    fig, axes = plt.subplots(nrows=2, ncols=2, squeeze=False,
                            sharex=False, sharey=False, figsize=(12, 6))
        

    ax_flux = axes[1,0]
    ax_rv = axes[1,1]

    new_theta = np.linspace(-180, 180, theta.shape[0]+1)[:-1]
    flux_data = rv_star.map.flux(theta=new_theta).eval()

    rv_data = rv_star.map.rv(theta=new_theta).eval()

    flux_image = ax_flux.plot(new_theta, flux_data,label="sample")
    ax_flux.set_xlabel("Angle of rotation [degrees]", fontsize=16)
    ax_flux.set_ylabel("Flux [normalized]", fontsize=16)
    ax_flux.yaxis.set_major_formatter(ticker.FuncFormatter(custom_formatter))
    ax_flux.axvline(x=0, color='gray', linestyle='--', linewidth=1)
    L_flux = ax_flux.legend(loc="upper left",handlelength=0)

    rv_image = ax_rv.plot(new_theta, rv_data,label="sample")
    ax_rv.set_xlabel("Angle of rotation [degrees]", fontsize=16)
    ax_rv.set_ylabel("Radial velocity [m/s]", fontsize=16)
    ax_rv.axvline(x=0, color='gray', linestyle='--', linewidth=1)
    L_rv = ax_rv.legend(loc="upper left",handlelength=0)

    plt.tight_layout()

    img1,image1,lonlines1,latlines1 = rv_star.map.show(theta=theta,rv=False, ax=axes[0,0],
                                                        show_image=False,colorbar=cbar_pos)
    
    rv_star.map.show(theta=theta,rv=True, ax=axes[0,1],show_image=True, colorbar=cbar_pos, 
                        norm=matplotlib.colors.CenteredNorm(),
                        extra_image=[img1,image1,lonlines1,latlines1],
                        extra_lines = [(flux_data,flux_image),(rv_data,rv_image)],
                        legend_list = [L_flux,L_rv],
                        file=fname, dpi = 300, transparent=transparent)



def create_rv(cfile):

    print("\nCalling create_eigens")
    print("----------------------------------------------------------------------------")
    eigeny, evalues, evectors, ecurves, lcs, star, cfg, fit = \
        create_eigens.create_eigens(cfile)
    
    print("Setting current directory")
    subdir = cfg.folder
    outdir = os.path.join(cfg.outdir, subdir)
    rv_path = os.path.join(outdir,"rv_outputs")

    if not os.path.isdir(rv_path):
        os.mkdir(rv_path)

    print("Creating a new star with rv=True")
    lmax = cfg.twod.lmax
    udeg = cfg.twod.udeg
    rv_star = utils.initstar(fit, lmax, udeg=udeg, include_rv=True)
    ncurves, ny = eigeny.shape

    theta = np.linspace(0, 360, 60)

    print("Looping through each map and generating plots")
    print("----------------------------------------------------------------------------")
    for i in range(ncurves):
        print(f"\tCreating plots for map_{i}")
        rv_star.map[1:,:] = 0

        yi = 1
        for l in range(1, lmax + 1):
            for m in range(-l, l + 1):
                rv_star.map[l, m] = eigeny[i,yi]
                yi += 1

        _, factor = adjust_star(rv_star,eigeny[i])
        
        print(f"\tMap {i} has been normalized with a factor of {factor}.")

        if not os.path.isdir(f"{rv_path}/map_{i}"):
            os.mkdir(f"{rv_path}/map_{i}")

        image = rv_star.map.render(projection="rect",rv=True).eval()

        rv_star.map.show(theta=theta,rv=False, file=f"{rv_path}/map_{i}/flux_map.mp4")
        rv_star.map.show(theta=theta,rv=True, file=f"{rv_path}/map_{i}/rv_map.mp4")
        
        rv_flux_map_ani(rv_star,theta,f"{rv_path}/map_{i}/both_maps.mp4")


        plt.figure(figsize=(12, 5))
        image = rv_star.map.render(projection="rect",rv=False).eval()
        plt.imshow(image, origin="lower", cmap="plasma", extent=(-180, 180, -90, 90))
        plt.xlabel("longitude [deg]")
        plt.ylabel("latitude [deg]")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(f"{rv_path}/map_{i}/flux_rect.png")
        plt.close()

        flux_rv_line(rv_star,theta,
                     flux_name=f"{rv_path}/map_{i}/flux_curve.png",rv_name=f"{rv_path}/map_{i}/rv_curve.png")

        multi_phase_plot(rv_star,fname=f"{rv_path}/map_{i}/map_slideshow.png")


        map_curve_ani(rv_star,theta,fname=f"{rv_path}/map_{i}/animated_plots.mp4")
        
        
        print(f"\tMap_{i} plots are complete!")
        print("\t----------------------------")

    print(f"All plots have been saved to {rv_path}!")
    print("----------------------------------------------------------------------------")



        







if __name__ == "__main__":
    """
    """
    # print(sys.argv) #Uncomment if you want to see command line arguments 
    if len(sys.argv) < 2:
        print("ERROR: Call structure is python create_rv.py <configuration file>.")
        sys.exit()
    else:
        cfile = sys.argv[1]

    create_rv(cfile)

    print("done")
    sys.exit()