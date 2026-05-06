"""
File to create visualization of all eigenmaps & their respective lightcurves
[insert citation for my paper and also github here]

Plots adapted from: https://github.com/rychallener/theresa
Challener, R. C., & Rauscher, E. 2022, AJ, 163, 117, doi: 10.3847/1538-3881/ac4885


(If theano shuts down and says 'constant folding error', see note in create_eigens.py)
"""

#general imports
import os
import sys
import starry2 as starry
import numpy as np
import matplotlib as mpl
mpl.rcParams['axes.formatter.useoffset'] = False
import matplotlib.pyplot as plt

# Directory structure
maindir    = os.path.dirname(os.path.realpath(__file__))
libdir     = os.path.join(maindir, 'lib')

# lib imports
sys.path.append(libdir)
from lib import plots
from lib.spotable import spotable as se


# un-spot-able imports
import create_eigens
import create_rv


# Set up starry configuration
starry.config.quiet = True
starry.config.lazy = True
sys.setrecursionlimit(10000) # starry seems to have a lot of recursion

# Whether or not to show print statements in code
dpm = True #change to False to make quiet

def set_emap_directory(fit):
    cfg = fit.cfg

    subdir = cfg.folder
    outdir = os.path.join(cfg.outdir, subdir)
    emaps_path = os.path.join(outdir,"emap_outputs")

    se(f"\tSetting directory to:\n\t\033[34m{emaps_path}\033[0m\n", dp = dpm)

    if not os.path.isdir(emaps_path):
        os.mkdir(emaps_path)

    return emaps_path

def create_emaps(star, eigeny, emaps_path=None, proj='rect', 
                 transparent=False, labels=False, individual=True,
                 colorbar=True, smooth=True):
    
    """
    Function generates various plots associated with the eigenmaps (emaps)  
    Can be set to have different projection styles  
    Will always create overall emap plot and is set by default to create individual as well

    Arguments
    ---------
    star: object
        A starry star object, initialized with a cfg file
    
    eigeny: 2D array
        (ncurves, ny) array of y coefficients for each harmonic. 
        ncurves is the number of harmonics, including only positive versions and excluding Y00: (lmax + 1)**2 - 1
        ny is the number of y coefficients to describe a harmonic with degree lmax: (lmax + 1)**2 (includes y00)

    emapth_path: string (optional)
        Path to folder where emap plots will be saved.  If None, will display image instead of saving

    proj: str (optional)
        Type of projection to use in 2D plots of eigenmaps

    *NOTE* Remaining agruments are optional and used to adjust final plot.  These are useful when using code
    to prepare a presentation/paper but not needed for general use

    transparent: boolean (optional)
        Whether to make plots transparent.  Default to False

    labels: boolean (optional)
        Wether to add axis and title labels to overall plot.  Default to False

    indiviudal: boolean (optional)
        If True, will generate individual emap plots along with overall.  Default is True
        Note, if individual is False, colorbar and smooth parameters do nothing 

    colorbar: boolean (optional)
        If True, will generate colorbar on individual emap plots.  Default is True

    smooth: boolean (optional)
        If True, will generate smoothed individual emap plots.  Default is True

    Returns
    -------
    None
    """    

    
    se("\tPlotting eigenmaps", dp = dpm)
    se("\t------------------------------------------------", dp = dpm)

    ncurves, ny = eigeny.shape

    se(f"\tSetting projection to {proj}", dp = dpm)

    if proj == 'ortho':
        extent = (-90, 90, -90, 90)
        fname = 'emaps-ortho.png'
    elif proj == 'rect':
        extent = (-180, 180, -90, 90)
        fname = 'emaps-rect.png'
    elif proj == 'moll':
        extent = (-180, 180, -90, 90)
        fname = 'emaps-moll.png'

    ncols = int(np.sqrt(ncurves) // 1)
    nrows = int(ncurves // ncols + (ncurves % ncols != 0))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False,
                             sharex=True, sharey=True)
    
    if individual:
        se(f"\n\tGenerating indivdiual emap plots", dp = dpm)
    
    for j in range(ncurves):
        star.map[:,:] = eigeny[j]

        xloc = j %  ncols
        yloc = j // ncols
        ax = axes[yloc, xloc]

        rendered_map = star.map.render(theta=0, projection=proj).eval()
        
        ax.imshow(rendered_map,
                  origin="lower",
                  cmap="plasma",
                  extent=extent)
        
        if individual:
            if emaps_path:
                indiv_path = os.path.join(emaps_path, f"{proj}_emaps")

                if not os.path.isdir(indiv_path):
                    os.mkdir(indiv_path)
                
                if smooth:
                    if proj == 'ortho':
                        temp_fig = plt.figure()
                    else:
                        temp_fig = plt.figure(figsize=(12, 5))
                    image = star.map.render(projection=proj).eval()
                    plt.imshow(image, origin="lower", cmap="plasma", extent=extent)
                    plt.xlabel("Longitude [deg]")
                    plt.ylabel("Latitude [deg]")
                    if colorbar:
                        plt.colorbar()
                    plt.tight_layout()
                    plt.savefig(f"{indiv_path}/map_{j}", dpi = 300,transparent=transparent)
                    plt.close(temp_fig)
                    pass
                else:
                    star.map.show(theta=0,projection=proj,file=f"{indiv_path}/map_{j}",
                              dpi=300,transparent=transparent, colorbar=colorbar)
            else:
                star.map.show(theta=0,projection=proj,
                              dpi=300,transparent=transparent, colorbar=colorbar)
            
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax.set_frame_on(False)

    if labels:
        fig.supxlabel("Longitude [deg]")
        fig.supylabel("Latitude [deg]")
        fig.suptitle(f"Eigenmaps - {proj.capitalize()} Projection")

    fig.tight_layout()

    se(f"\n\tGenerating overall emap plot", dp = dpm)
    
    
    if emaps_path:
        plt.savefig(os.path.join(emaps_path, fname), transparent=transparent, dpi=300)
    else:
        plt.show()
    
    plt.close(fig)

    # Resetting star to be uniform map
    star.map[0,0] = 1
    star.map[1:,:] = 0

    se("\t------------------------------------------------", dp = dpm)

    return None

def create_eflux(star, eigeny, emaps_path=None,
                 transparent=False, labels=False, individual=True):

    """
    Function generates overall and individual light curve (lc) plots for each eigenmap (emap)  
    Will always create overall lc plot and is set by default to create individual as well

    Arguments
    ---------
    star: object
        A starry star object, initialized with a cfg file
    
    eigeny: 2D array
        (ncurves, ny) array of y coefficients for each harmonic. 
        ncurves is the number of harmonics, including only positive versions and excluding Y00: (lmax + 1)**2 - 1
        ny is the number of y coefficients to describe a harmonic with degree lmax: (lmax + 1)**2 (includes y00)

    emapth_path: string (optional)
        Path to folder where emap plots will be saved.  If None, will display image instead of saving

    *NOTE* Remaining agruments are optional and used to adjust final plot.  These are useful when using code
    to prepare a presentation/paper but not needed for general use

    transparent: boolean (optional)
        Whether to make plots transparent.  Default to False

    labels: boolean (optional)
        Wether to add axis and title labels to overall plot.  Default to False

    indiviudal: boolean (optional)
        If True, will generate individual emap plots along with overall.  Default is True.
        Note, if individual is False, colorbar and smooth parameters do nothing 

    Returns
    -------
    None
    """    

    
    se("\tPlotting light curves", dp = dpm)
    se("\t------------------------------------------------", dp = dpm)

    ncurves, ny = eigeny.shape

    ncols = int(np.sqrt(ncurves) // 1)
    nrows = int(ncurves // ncols + (ncurves % ncols != 0))

    theta = np.linspace(0, 360, 360)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False,
                             sharex=True, sharey=False)
    
    if individual:
        se(f"\tGenerating indivdiual light curve plots\n", dp = dpm)
    
    for j in range(ncurves):
        star.map[:,:] = eigeny[j]

        xloc = j %  ncols
        yloc = j // ncols
        ax = axes[yloc, xloc]
        
        ax.plot(theta, star.map.flux(theta=theta).eval())
        ax.axis('off')

        if individual:
            if emaps_path:
                indiv_path = os.path.join(emaps_path, f"rlc")

                if not os.path.isdir(indiv_path):
                    os.mkdir(indiv_path)
                
                create_rv.flux_rv_line(star,theta,flux_name=f"{indiv_path}/rlc_{j}",flux_only=True)

            else:
                create_rv.flux_rv_line(star,theta,flux_name=None,flux_only=True)

    se(f"\tGenerating overall light curve plot", dp = dpm)

    if labels:
        fig.supxlabel("Longitude [deg]")
        fig.supylabel("Latitude [deg]")
        fig.suptitle(f"Eigenmaps - Rotational Light Curves")
    
    fig.tight_layout()
    plt.savefig(f"{emaps_path}/all_rlc", transparent=transparent, dpi=300)
    plt.close(fig)

    # Resetting star to be uniform map
    star.map[0,0] = 1
    star.map[1:,:] = 0

    se("\t------------------------------------------------", dp = dpm)

    return None






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
    
    se("\n\033[32mCreating emap visualizations:\033[0m", dp = dpm)
    se("----------------------------------------------------------------------------", dp = dpm)
    
    emaps_path = set_emap_directory(fit)


    create_emaps(star, eigeny, emaps_path)

    se("----------------------------------------------------------------------------", dp = dpm)

    se("\n\033[32mCreating light curve visualizations:\033[0m", dp = dpm)
    se("----------------------------------------------------------------------------", dp = dpm)

    create_eflux(star, eigeny, emaps_path)

    se("----------------------------------------------------------------------------", dp = dpm)

    sys.exit("\033[32mdone\033[0m")

