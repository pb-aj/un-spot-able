"""
[Description needed here]
https://github.com/pb-aj/un-spot-able
"""

#general imports
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import starry2 as starry
import faulthandler
faulthandler.enable()

# Directory structure
maindir    = os.path.dirname(os.path.realpath(__file__))
libdir     = os.path.join(maindir, 'lib')

# Lib imports
sys.path.append(libdir)
from lib import utils
from lib.spotable import spotable as se


# py imports
import create_eigens
import create_emaps

# Set up starry configuration
starry.config.quiet = True
starry.config.lazy = True
sys.setrecursionlimit(10000) # starry seems to have a lot of recursion


# Whether or not to show print statements in code
dpm = True #change to False to make quiet



def set_rv_directory(fit):
    """
    Function generates various plots associated with the eigenmaps (emaps)  
    Can be set to have different projection styles  
    Will always create overall emap plot and is set by default to create individual as well

    Arguments
    ---------
    fit: fit class (defined in lib/fitclass.py)
        stored fit values read from cfg

    Returns
    -------
    rv_path : string
        path to save eigenmap (emap) plots to
    """

    cfg = fit.cfg

    subdir = cfg.folder
    outdir = os.path.join(cfg.outdir, subdir)
    rv_path = os.path.join(outdir,"rv_outputs")

    se(f"\tSetting directory to:\n\t\033[34m{rv_path}\033[0m\n", dp = dpm)

    if not os.path.isdir(rv_path):
        os.mkdir(rv_path)

    return rv_path

def adjust_star(star, lower_limit=0, uni_comp = None, iterator = .1):
    """
    Adjust star to remove negative flux values by scaling uniform component.
    Exact lower limit and iterator can be changed depending on desired accurary

    Arguments
    ---------
    star: object
        A starry star object, initialized with a cfg file

    lower_limit: float (optional)
        Minimum allowed flux on stellar map.  Default is 0

    uni_comp: float (optional)
        Starting uniform component (Y00) for star.  Default is None
        If default used, will take Y00 from star

    iterator: float (optional)
        Amount to increase Y00 after each check.  Default is .1

    Returns
    -------
    None
    """
    if uni_comp:
        star.map[0,0] = uni_comp

    min_val = star.map.minimize()[-1].eval()

    while min_val <= lower_limit:

            star.map[0,0] += iterator

            min_val = star.map.minimize()[-1].eval()
            

    return None


def rv_flux_map_ani(rv_star, theta, fname=None, interval=75, colorbar = 'bottom', transparent = True):
    """
    Generate an animated ortho projection of flux and RV side by side
    Does not generate the respective light or RV curves

    Arguments
    ---------
    rv_star: object
        A starry star object (w/rv=True), initialized with a cfg file

    theta: 1D array
        Degrees to evalulate and animate rv and flux at

    fname: string (optional)
        Name of file to save animation to.  Default is None
        If None, will display image instead of saving

    interval: int (optional)
        Delay between frames in milliseconds (see matplotlib.animation.FuncAnimation)
        Default is 75

    colorbar: boolean or "bottom" (optional)
        Whether to include a color bar or not.  Default is False
        When True colorbar is on right
        Using "bottom" places the colorbar on the bottom 

    transparent: boolean (optional)
        Whether to make plots transparent.  Default to False
        *NOTE* This will only work if passed filename ends in .gif (transparent not easily supported by .mp4)

    Returns
    -------
    None
    """

    fig, axes = plt.subplots(nrows=1, ncols=2, squeeze=False,
                                    sharex=False, sharey=False, figsize=(12, 6))
        
    img1,image1,lonlines1,latlines1 = rv_star.map.show(theta=theta,rv=False, 
                                                       ax=axes[0,0],show_image=False, 
                                                       colorbar=colorbar, transparent=transparent)
    
    rv_star.map.show(theta=theta,rv=True, ax=axes[0,1],show_image=True, 
                        extra_image=[img1,image1,lonlines1,latlines1],file=fname, 
                        dpi = 300,interval=interval,colorbar=colorbar, transparent=transparent)
    
    plt.close(fig)

def flux_rv_line(rv_star,theta,flux_name=None,rv_name=None,flux_only=False,rv_only=False):
    """
    [desc]

    Arguments
    ---------
    x: type (optional)
        [desc]

    Returns
    -------
    x: type
        [desc]
    """

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
    """
    [desc]

    Arguments
    ---------
    x: type (optional)
        [desc]

    Returns
    -------
    x: type
        [desc]
    """

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
    plt.close(fig)


def map_curve_ani(rv_star,theta,fname=None,cbar_pos="bottom",transparent=False,interval=75):
    """
    [desc]

    Arguments
    ---------
    x: type (optional)
        [desc]

    Returns
    -------
    x: type
        [desc]
    """

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
                        file=fname, dpi = 300, transparent=transparent, interval=interval)



def create_rv(eigeny, fit, theta = np.linspace(0, 360, 360)):
    """
    [desc]

    Arguments
    ---------
    x: type (optional)
        [desc]

    Returns
    -------
    x: type
        [desc]
    """

    cfg = fit.cfg

    se("\tCreating a new star with rv=True\n",dp=dpm)

    lmax = cfg.twod.lmax
    udeg = cfg.twod.udeg
    rv_star = utils.initstar(fit, lmax, udeg=udeg, include_rv=True)
    ncurves, ny = eigeny.shape

    ani_interval = int(6000 / len(theta))

    se("\tLooping through each map and generating plots", dp=dpm)
    se("\t------------------------------------------------", dp = dpm)
    for i in range(ncurves):
        se(f'\t\t\u2022 Adjusting "Emap {i}" to have only positive flux', dp=dpm)
        rv_star.map[:,:] = eigeny[i]

        adjust_star(rv_star)
        
        se(f'\t\t\u2022 "Emap {i}" has been normalized with a factor of {rv_star.map.y.eval()[0]:.1f}', dp=dpm)

        se(f'\t\t\u2022 Creating plots for "Emap {i}"', dp=dpm)

        indiv_path = f"{rv_path}/map_{i}"

        if not os.path.isdir(indiv_path):
            os.mkdir(indiv_path)

        rv_star.map.show(theta=theta,rv=False, file=f"{indiv_path}/flux_map.mp4", interval = ani_interval)
        rv_star.map.show(theta=theta,rv=True, file=f"{indiv_path}/rv_map.mp4", interval = ani_interval)
        
        rv_flux_map_ani(rv_star,theta,f"{indiv_path}/both_maps.mp4",interval= ani_interval)
        
        create_emaps.emap_plot(rv_star, indiv_path=indiv_path, proj='rect', other_fname=None, 
                 transparent=False, colorbar=True, smooth=True)

        flux_rv_line(rv_star,theta,
                     flux_name=f"{rv_path}/map_{i}/flux_curve.png",rv_name=f"{rv_path}/map_{i}/rv_curve.png")

        multi_phase_plot(rv_star,fname=f"{rv_path}/map_{i}/map_slideshow.png")


        map_curve_ani(rv_star,theta,fname=f"{rv_path}/map_{i}/animated_plots.mp4", interval = ani_interval)
        
        
        se(f'\t\t\u2022 "Emap {i}" plots are complete!', dp=dpm)
        se("\t\t------------------------------------------------", dp=dpm)

    se(f"\tAll plots have been saved to:\n\t\033[34m{rv_path}\033[0m", dp = dpm)
    se("----------------------------------------------------------------------------",dp = dpm)








if __name__ == "__main__":
    # Uncomment if you want to see command line arguments 
    # se(sys.argv,dp = dpm) 

    # Check command line input is correct
    if len(sys.argv) < 2:
        se("\n----------------------------------------------------------------------------",dp = dpm)
        se('\033[31mERROR:\033[0m' + ' Call structure is "\033[34mpython create_eigens.py <configuration file>\033[0m"',dp = dpm)
        se("----------------------------------------------------------------------------",dp = dpm)
        sys.exit()
    else:
        cfile = sys.argv[1]

    se("\n\033[32mCalling create_eigens:\033[0m", dp = dpm)
    se("----------------------------------------------------------------------------", dp = dpm)
    eigeny, evalues, evectors, ecurves, lcs, star, fit = \
        create_eigens.create_eigens(cfile,prompt_user=False)
    
    se("\n\033[32mCalculating & Visualizing RV:\033[0m", dp = dpm)
    se("----------------------------------------------------------------------------", dp = dpm)

    rv_path = set_rv_directory(fit)

    create_rv(eigeny, fit)

    sys.exit("\033[32mdone\033[0m")