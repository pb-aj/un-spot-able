"""
File to create visualization of all eigenmaps & their respective lightcurves
https://github.com/pb-aj/un-spot-able

Plots adapted from: https://github.com/rychallener/theresa
Challener, R. C., & Rauscher, E. 2022, AJ, 163, 117, doi: 10.3847/1538-3881/ac4885
"""

#general imports
import os
import sys
import starry2 as starry
import numpy as np
import matplotlib as mpl
mpl.rcParams['axes.formatter.useoffset'] = False
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import faulthandler
faulthandler.enable()

# Directory structure
maindir    = os.path.dirname(os.path.realpath(__file__))
libdir     = os.path.join(maindir, 'lib')

# lib imports
sys.path.append(libdir)
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
    emaps_path : string
        path to save eigenmap (emap) plots to
    """
    cfg = fit.cfg

    subdir = cfg.folder
    outdir = os.path.join(cfg.outdir, subdir)
    emaps_path = os.path.join(outdir,"emap_outputs")

    se(f"\tSetting directory to:\n\t\033[34m{emaps_path}\033[0m\n", dp = dpm)

    if not os.path.isdir(emaps_path):
        os.mkdir(emaps_path)

    return emaps_path

def emap_plot(star, indiv_path=None, proj='rect', other_fname=None, 
                 transparent=False, colorbar=True, smooth=True, fontsize=16,
                 labels=True, border=True, ticks=True, gridlines=False):
    
    """
    Function generates a single emap projection depending on passed star
    Can be set to have different projection styles 

    Arguments
    ---------
    star: object
        A starry star object, initialized with a cfg file

    indiv_path: string (optional)
        Path to folder where emap plot will be saved.  If None, will display image instead of saving

    proj: str (optional)
        Type of projection to use in 2D plots of eigenmaps

    *NOTE* Remaining agruments are optional and used to adjust final plot.  These are useful when using code
    to prepare a presentation/paper but not needed for general use

    transparent: boolean (optional)
        Whether to make plots transparent.  Default to False

    colorbar: boolean (optional)
        If True, will generate colorbar on individual emap plots.  Default is True

    smooth: boolean (optional)
        If True, will generate smoothed individual emap plots.  Default is True

    *NOTE* Remaining arguments only have an effect if smooth=True as starry does not have these options by default

    fontsize: int (optional)
        Sets size of axis labels and title (1.5x axis).  Default is 24

    label: boolean (optional)
        If True, will generate axis and title labels.  Default is True

    border: boolean (optional)
        Wether to add border around each axis plot.  Default to True

    tick: boolean (optional)
        If True, will include ticks and values.  Default is True
        Only works for proj="rect"

    gridlines: boolean (optional)
        If True, will include grid lines.  Default is False
        Only works for proj="rect"

    Returns
    -------
    None
    """

    if proj == 'ortho':
        extent = (-90, 90, -90, 90)
        fname = 'emap-ortho.png'
    elif proj == 'rect':
        extent = (-180, 180, -90, 90)
        fname = 'emap-rect.png'
    elif proj == 'moll':
        extent = (-180, 180, -90, 90)
        fname = 'emap-moll.png'

    if other_fname:
        fname = other_fname

    if smooth:

        if proj == 'ortho':
            temp_fig = plt.figure(figsize=(7,5))
        else:
            temp_fig = plt.figure(figsize=(12, 5))
        image = star.map.render(projection=proj,rv=False).eval()
        plt.imshow(image, origin="lower", cmap="plasma", extent=extent)

        if colorbar:
            plt.colorbar()

        if labels:

            if proj == "rect":
                plt.xlabel("Longitude [deg]",fontsize=fontsize)
                plt.ylabel("Latitude [deg]",fontsize=fontsize)

            if other_fname:
                plt.title(f"Eigenmap {other_fname.split('_')[-1]} - Rectangular Projection",fontsize=int(fontsize*1.5))
            else:
                plt.title("Eigenmap - Rectangular Projection",fontsize=int(fontsize*1.5))


        if gridlines and proj == "rect":
            plt.grid(color="k",linestyle=":")

        if not ticks or proj != "rect":
            plt.gca().set_xticklabels([])
            plt.gca().set_yticklabels([])
            plt.tick_params(left=False, bottom=False)

        if not border:
            plt.gca().set_frame_on(False)
        
        plt.tight_layout()
        
        if indiv_path:
            plt.savefig(f"{indiv_path}/{fname}", dpi = 300, transparent=transparent)
        else:
            plt.show()
        plt.close(temp_fig)
        pass
    else:
        if indiv_path:
            star.map.show(theta=0,projection=proj,rv=False,file=f"{indiv_path}/{fname}",
                    dpi=300,transparent=transparent, colorbar=colorbar)
        else:
            star.map.show(theta=0,projection=proj,rv=False,
                        dpi=300,transparent=transparent, colorbar=colorbar)
        

def create_emaps(star, eigeny, emaps_path=None, proj='rect', 
                 transparent=False, a_labels=False, a_border=False, 
                 a_ticks=False, a_gridlines=False, fontsize=16,
                 individual=True, colorbar=True, smooth=True):
    
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

    a_labels: boolean (optional)
        Wether to add axis and title labels to overall plot (a represents all).  Default to False

    a_border: boolean (optional)
        Wether to add border around each axis plot on overall plot.  Default to False

    a_ticks: boolean (optional)
        If True, will add ticks and values to overall plot.  Default is False

    a_gridlines: boolean (optional)
        If True, will add grid lines to overall plot.  Default is False

    fontsize: int (optional)
        Sets size of axis labels and title (1.5x axis).  Default is 24

    indiviudal: boolean (optional)
        If True, will generate individual emap plots along with overall.  Default is True
        Will not generate if no path set to avoid clutter in image display.
        
    *NOTE* If individual is False, colorbar and smooth parameters do nothing 

    colorbar: boolean (optional)
        If True, will generate colorbar on individual emap plots.  Default is True

    smooth: boolean (optional)
        If True, will generate smoothed individual emap plots.  Default is True

    Returns
    -------
    None
    """    

    
    se("\tPlotting eigenmaps:", dp = dpm)
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
                             sharex=True, sharey=True, figsize=(7,5))
    
    if individual and emaps_path:
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
        
        if individual and emaps_path:
            indiv_path = os.path.join(emaps_path, f"{proj}_emaps")

            if not os.path.isdir(indiv_path):
                os.mkdir(indiv_path)

            emap_plot(star, indiv_path=indiv_path, proj=proj, other_fname=f"emap_{j}", 
                 transparent=transparent, colorbar=colorbar, smooth=smooth)


        if not a_ticks:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.tick_params(left=False, bottom=False)
        
        if not a_border:
            ax.set_frame_on(False)

        if a_gridlines:
            ax.grid(color="k",linestyle=":")

    if a_labels:
        fig.supxlabel("Longitude [deg]",fontsize=fontsize)
        fig.supylabel("Latitude [deg]",fontsize=fontsize)
        fig.suptitle(f"Eigenmaps - {proj.capitalize()} Projection",fontsize=int(fontsize*1.5))

    fig.tight_layout()

    se(f"\n\tGenerating overall emap plot", dp = dpm)
    
    
    if emaps_path:
        plt.savefig(os.path.join(emaps_path, fname), transparent=transparent, dpi=300)
    else:
        plt.show()

    # Resetting star to be uniform map
    star.map[0,0] = 1
    star.map[1:,:] = 0

    se("\t------------------------------------------------", dp = dpm)

    plt.close('all')

    return None


def create_eflux(star, eigeny, emaps_path=None, theta = np.linspace(0, 360, 360),
                 transparent=False, fontsize=16, a_gridlines=False,
                 a_labels=False, a_border=False, a_ticks=False,
                 individual=True):

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

    theta: 2D array (optional)
        Degrees to evaluate light curve at for each emap.  Default is every degree - np.linspace(0, 360, 360)

    *NOTE* Remaining agruments are optional and used to adjust final plot.  These are useful when using code
    to prepare a presentation/paper but not needed for general use

    transparent: boolean (optional)
        Whether to make plots transparent.  Default to False

    fontsize: int (optional)
        Sets size of axis labels and title (1.5x axis).  Default is 24

    a_gridlines: boolean (optional)
        If True, will add grid lines to overall plot.  Default is False

    a_labels: boolean (optional)
        Wether to add axis and title labels to overall plot (a represents all).  Default to True

    a_border: boolean (optional)
        Wether to add border around each axis plot on overall plot.  Default to False

    a_ticks: boolean (optional)
        If True, will add ticks and values to overall plot.  Default is False

    indiviudal: boolean (optional)
        If True, will generate individual emap plots along with overall.  Default is True
        Will not generate if no path set to avoid clutter in image display.

    Returns
    -------
    None
    """    

    
    se("\tPlotting light curves:", dp = dpm)
    se("\t------------------------------------------------", dp = dpm)

    ncurves, ny = eigeny.shape

    ncols = int(np.sqrt(ncurves) // 1)
    nrows = int(ncurves // ncols + (ncurves % ncols != 0))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False,
                             sharex=True, sharey=False, figsize=(10, 6))
    
    if individual and emaps_path:
        se(f"\tGenerating indivdiual light curve plots\n", dp = dpm)

    if a_ticks:
        t1 = theta.max()//6
        x_tick_values = [t1, t1*5]
    
    for j in range(ncurves):
        star.map[:,:] = eigeny[j]

        xloc = j %  ncols
        yloc = j // ncols
        ax = axes[yloc, xloc]

        j_flux = star.map.flux(theta=theta).eval()
        
        ax.plot(theta, j_flux)

        if individual and emaps_path:
            indiv_path = os.path.join(emaps_path, f"rlc")

            if not os.path.isdir(indiv_path):
                os.mkdir(indiv_path)
            
            create_rv.flux_rv_line(star,theta,flux=j_flux,flux_name=f"{indiv_path}/rlc_{j}",flux_only=True)

        if not a_ticks:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.tick_params(left=False, bottom=False)
        else:
            
            ax.set_xticks(x_tick_values)
            ax.set_xticklabels([f"{val:.0f}" for val in x_tick_values])

            min_f = j_flux.min()
            max_f = j_flux.max()
            if max_f - min_f > 1e-10:
                y_tick_values = [min_f,max_f] 

            else:
                y_tick_values = [max_f]
            ax.set_yticks(y_tick_values)
            ax.set_yticklabels([f"{val:.2g}" for val in y_tick_values])

        
        if not a_border:
            ax.set_frame_on(False)

        if a_gridlines:
            ax.grid(color="k",linestyle=":")

    se(f"\tGenerating overall light curve plot", dp = dpm)

    if a_labels:
        fig.supxlabel("Angle of rotation [degrees]",fontsize=fontsize)
        fig.supylabel("Flux [normalized]",fontsize=fontsize)
        fig.suptitle(f"Eigenmaps - Rotational Light Curves",fontsize=int(fontsize*1.5))
    
    fig.tight_layout()

    if emaps_path:
        plt.savefig(f"{emaps_path}/all_rlc", transparent=transparent, dpi=300)
    else:
        plt.show()

    # Resetting star to be uniform map
    star.map[0,0] = 1
    star.map[1:,:] = 0

    se("\t------------------------------------------------", dp = dpm)

    plt.close('all')

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


    create_emaps(star, eigeny, emaps_path=emaps_path)

    se("----------------------------------------------------------------------------", dp = dpm)

    se("\n\033[32mCreating light curve visualizations:\033[0m", dp = dpm)
    se("----------------------------------------------------------------------------", dp = dpm)

    create_eflux(star, eigeny, emaps_path=emaps_path)

    se("----------------------------------------------------------------------------", dp = dpm)

    sys.exit("\033[32mdone\033[0m")

