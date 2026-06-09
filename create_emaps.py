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
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from cmcrameri import cm
import faulthandler
faulthandler.enable()

# Directory structure
maindir    = os.path.dirname(os.path.realpath(__file__))
libdir     = os.path.join(maindir, 'lib')

# lib imports
sys.path.append(libdir)
from lib.spotable import spotable as se
from lib import lat_lon_lines


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

def emap_plot(star, indiv_path=None, proj='moll', other_fname=None, cmap = cm.bam, 
                 center_flux=0, cmap_norm = None,
                 transparent=False, colorbar=True, colorbar_label = True, fontsize=16,
                 labels=True, title=None, border=True, ticks=True, gridlines=True, 
                 gridcolor="k", unseen_line=True, cover_unseen=True):
    
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

    cmap: str (optional)
        What color map to use in plots.  Default is cm.bam
        To use the original starry colors, set to 'plasma'
    
    center_flux: float (optional)
        Value to center colorbar at.  Default is 0

    cmap_norm: Matplotlib Normalization (optional)
        Normalization to use for map, if None uses CenteredNorm around center_flux value.  Default is None

    transparent: boolean (optional)
        Whether to make plots transparent.  Default to False

    colorbar: boolean (optional)
        If True, will generate colorbar on individual emap plots.  Default is True

    colorbarlabel: boolean (optional)
        If True, will generate a label for the colorbar on individual emap plots.  Default is True

    fontsize: int (optional)
        Sets size of axis labels and title (1.5x axis).  Default is 24

    label: boolean (optional)
        If True, will generate axis labels.  Default is True

    title: str (optional)
        If value passed, will set it as the title.  Default is None

    border: boolean (optional)
        Wether to add border around each axis plot.  Default to True

    tick: boolean (optional)
        If True, will include ticks and values.  Default is True
        Only works for proj="rect"

    gridlines: boolean (optional)
        If True, will include grid lines.  Default is False
        Only works for proj="rect"

    gridcolor: string (optional)
        Color to set curve to be (see https://matplotlib.org/stable/gallery/color/named_colors.html).  
        Default is 'darkgrey'.

    unseen_line: boolean (optional)
        Whether to include gridline that shows what is never visible according to star's inclination.
        Default is False.  Only applies if gridlines=True

    cover_unseen: boolean (optional)
        Whether to black out region of star that is never visibale according to star's inclination.
        Default is True.

    Returns
    -------
    None
    """

    if proj == "ortho180":
        image = star.map.render(theta=180,projection=proj,rv=False).eval()
    else:
        image = star.map.render(theta=0,projection=proj,rv=False).eval()

    if proj == 'ortho':
        dx = 180.0 / image.shape[1] 
        dy = 180.0 / image.shape[0]

        extent = np.array([
            -90 - dx,  # Left (padded by one pixel width)
            90,       # Right
            -90 - dy,   # Bottom (padded by one pixel height)
            90         # Top
        ])

        fname = 'emap-ortho.png'

    elif proj == "ortho180":
        dx = 180.0 / image.shape[1] 
        dy = 180.0 / image.shape[0]

        extent = np.array([
            -90 - dx,  # Left (padded by one pixel width)
            90,       # Right
            -90 - dy,   # Bottom (padded by one pixel height)
            90         # Top
        ])

        fname = 'emap-ortho180.png'

    elif proj == 'rect':
        extent = (-180, 180, -90, 90)
        fname = 'emap-rect.png'
    elif proj == 'moll':
        dx = 360.0 / image.shape[1] 
        dy = 180.0 / image.shape[0]


        extent = np.array([
            -180 - dx/2,  # Left (padded by one pixel width)
            180 - dx/2,       # Right
            -90 - dy/2,   # Bottom (padded by one pixel height)
            90 - dy/2         # Top
        ])

        fname = 'emap-moll.png'

    if other_fname:
        fname = other_fname


    if proj == 'ortho' or proj == "ortho180":
        temp_fig = plt.figure(figsize=(7,5))
    else:
        temp_fig = plt.figure(figsize=(12, 5))

    if cmap_norm:
        plt.imshow(image, origin="lower", cmap=cmap, extent=extent,
                    norm=cmap_norm)
    else:
        plt.imshow(image, origin="lower", cmap=cmap, extent=extent,
                    norm=mpl.colors.CenteredNorm(vcenter=center_flux))

    if colorbar:
        cbar = plt.colorbar(aspect = 40, pad = .03, shrink = 0.95)
        if colorbar_label:
            cbar.set_label("Flux [Normalized]", size=12, rotation = 270, labelpad = 15)

    if labels:
        plt.xlabel("Longitude [deg]",fontsize=fontsize)
        plt.ylabel("Latitude [deg]",fontsize=fontsize)

    if title:
        plt.title(title,fontsize=int(fontsize*1.5))


    if gridlines: 
        if not cover_unseen and unseen_line:
            unseen_degree = -1 * star.map.inc.eval()
        else:
            unseen_degree = None

        if proj == "rect":
            plt.grid(color=gridcolor,linestyle=":")
            if not unseen_degree is None:
                plt.axhline(unseen_degree, c="k", ls="-", lw=4, alpha=1, zorder=1)
                plt.axhline(unseen_degree, c="deepskyblue", ls="-", lw=3, alpha=1, zorder=1)

        elif proj == "moll":

            lats = lat_lon_lines.get_moll_latitude_lines(unseen_deg=unseen_degree)
            latlines = [None for n in lats]

            for n, l in enumerate(lats):
                
                if unseen_degree and n == len(lats) - 1:

                    black_line = plt.plot(
                        l[0], l[1], c="k",ls="-", lw=4, alpha=1, zorder=1
                    )[0]
                    sky_line = plt.plot(
                        l[0], l[1], c="deepskyblue",ls="-", lw=3, alpha=1, zorder=1
                    )[0]
                else:
                    (latlines[n],) = plt.plot(
                        l[0], l[1], "k:", lw=0.5, alpha=1, zorder=0
                    )

            lons = lat_lon_lines.get_moll_longitude_lines(dlon=45)
            lonlines = [None for n in lons]
            for n, l in enumerate(lons):
                if (n == 0 or n == len(lons) - 1) and border:
                    (lonlines[n],) = plt.plot(
                        l[0], l[1], "k-", lw=1.5, alpha=1, zorder=1
                    )
                else:
                    (lonlines[n],) = plt.plot(
                        l[0], l[1], "k:", lw=0.5, alpha=1, zorder=0
                    )


            # Force the unseen_line to stay strictly inside the outer boundary line
            if unseen_degree and 'sky_line' in locals():

                left_path = lonlines[0].get_path()
                right_path = lonlines[-1].get_path()

                combined_vertices = list(left_path.vertices) + list(right_path.vertices[::-1])
                closed_oval_path = mpath.Path(combined_vertices)

                clip_patch = mpatches.PathPatch(closed_oval_path, transform=plt.gca().transData)
    
                sky_line.set_clip_path(clip_patch)
                black_line.set_clip_path(clip_patch)

            plt.ylim(-93,93)
            plt.xlim(-183,183)

        else:

            lats = lat_lon_lines.get_ortho_latitude_lines(unseen_deg=unseen_degree)
            latlines = [None for n in lats]

            for n, l in enumerate(lats):

                if unseen_degree and n == len(lats) - 1:

                    black_line = plt.plot(
                        l[0], l[1], c="k",ls="-", lw=3, alpha=1, zorder=1
                    )[0]
                    sky_line = plt.plot(
                        l[0], l[1], c="deepskyblue",ls="-", lw=2, alpha=1, zorder=1
                    )[0]

                else:
                    (latlines[n],) = plt.plot(
                        l[0], l[1], "k:", lw=0.5, alpha=1, zorder=0
                    )

            lons = lat_lon_lines.get_ortho_longitude_lines()
            lonlines = [None for n in lons]
            for n, l in enumerate(lons):
                if n == 0 or n == (len(lons) - 1) and border:
                    (lonlines[n],) = plt.plot(
                    l[0], l[1], "k-", lw=1.5, alpha=1, zorder=1
                    )
                else:
                    (lonlines[n],) = plt.plot(
                    l[0], l[1], "k:", lw=.5, alpha=1, zorder=0
                    )

            # Force the unseen_line to stay strictly inside the outer boundary line
            if unseen_degree and 'sky_line' in locals():

                left_path = lonlines[0].get_path()
                right_path = lonlines[-1].get_path()

                combined_vertices = list(left_path.vertices) + list(right_path.vertices[::-1])
                closed_oval_path = mpath.Path(combined_vertices)

                clip_patch = mpatches.PathPatch(closed_oval_path, transform=plt.gca().transData)
    
                sky_line.set_clip_path(clip_patch)
                black_line.set_clip_path(clip_patch)
                



            plt.xlim(-95,95)
            plt.ylim(-95,95)


    if cover_unseen:
        unseen_degree = -1 * star.map.inc.eval()

        if proj == "rect":

            plt.fill_between(np.linspace(-180,180,361), -90, unseen_degree, color="k", alpha = .6, edgecolor='none')

            pass

        elif proj == "moll":

            lats = lat_lon_lines.get_moll_latitude_lines(dlat=0, unseen_deg=unseen_degree)

            x_vals = np.array(lats[0][0])
            fill_val = np.array(lats[0][1])

            lat_xs = x_vals[~np.isnan(fill_val)]  

            lons = lat_lon_lines.get_moll_longitude_lines(dlon=360)
            lonlines = [None for n in lons]

            lon_xs = np.array([])
            lon_ys = np.array([])

            for n, l in enumerate(lons):
                x_is = np.argsort(l[0])

                lon_xs = np.append(lon_xs,l[0][x_is])
                lon_ys = np.append(lon_ys,l[1][x_is])

            x_mask = (lon_xs > np.min(lat_xs)) & (lon_xs < np.max(lat_xs))

            y_mask = (lon_ys < np.nanmedian(fill_val))

            xs = lon_xs[x_mask & y_mask]

            ys = lon_ys[x_mask & y_mask]

            plt.fill_between(xs, ys, np.nanmedian(fill_val), 
                    color="k", alpha = .6, edgecolor='none')

        else:
            lats = lat_lon_lines.get_ortho_latitude_lines(dlat=0, unseen_deg=unseen_degree)

            x_vals = np.array(lats[0][0])
            fill_val = np.array(lats[0][1])

            lat_xs = x_vals[~np.isnan(fill_val)]  

            lons = lat_lon_lines.get_ortho_longitude_lines(dlon=180)
            lonlines = [None for n in lons]

            lon_xs = np.array([])
            lon_ys = np.array([])

            for n, l in enumerate(lons):
                x_is = np.argsort(l[0])

                lon_xs = np.append(lon_xs,l[0][x_is])
                lon_ys = np.append(lon_ys,l[1][x_is])

            x_mask = (lon_xs > np.min(lat_xs)) & (lon_xs < np.max(lat_xs))

            y_mask = (lon_ys < np.nanmedian(fill_val))

            xs = lon_xs[x_mask & y_mask]

            ys = lon_ys[x_mask & y_mask]

            plt.fill_between(xs, ys, np.nanmedian(fill_val), 
                    color="k", alpha = .6, edgecolor='none')

    if not ticks or proj != "rect":
        plt.gca().set_xticklabels([])
        plt.gca().set_yticklabels([])
        plt.tick_params(left=False, bottom=False)
    else:
        plt.gca().set_xticks([-135,-90,-45,0,45,90,135])
        plt.gca().set_xticklabels([-135,-90,-45,0,45,90,135])

        plt.gca().set_yticks([-60,-30,0,30,60])
        plt.gca().set_yticklabels([-60,-30,0,30,60])

        plt.tick_params(direction="in")

    if not border or proj != "rect":
        plt.gca().set_frame_on(False)
    
    plt.tight_layout()
    
    if indiv_path:
        plt.savefig(f"{indiv_path}/{fname}", dpi = 300, transparent=transparent)
    else:
        plt.show()

    plt.close(temp_fig)

        

def create_emaps(star, eigeny, emaps_path=None, proj='moll', cmap = cm.bam, center_flux=0,
                 transparent=False, labels=True, title = None, border=True, 
                 ticks=True, gridlines=True, gridcolor="k", unseen_line=True, cover_unseen = True,
                 fontsize=16, standard_color = True, colorbar=True, colorbar_label = True, 
                 individual = True, standard_indiv = True):
    
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

    cmap: str (optional)
        What color map to use in plots.  Default is cm.bam
        To use the original starry colors, set to 'plasma'
    
    center_flux: float (optional)
        Value to center colorbar at for each plot.  Default is 0

    transparent: boolean (optional)
        Whether to make plots transparent.  Default to False

    labels: boolean (optional)
        Wether to add axis labels to overall plot (a represents all).  Default to False

    title: str (optional)
        If value passed, will set it as the title.  Default is None

    border: boolean (optional)
        Wether to add border around each axis plot on overall plot.  Default to False

    ticks: boolean (optional)
        If True, will add ticks and values to overall plot.  Default is False

    gridlines: boolean (optional)
        If True, will add grid lines to overall plot.  Default is False

    gridcolor: string (optional)
        Color to set curve to be (see https://matplotlib.org/stable/gallery/color/named_colors.html).  
        Default is 'darkgrey'.

    unseen_line: boolean (optional)
        Whether to include gridline that shows what is never visible according to star's inclination.
        Default is False.  Only applies if gridlines=True

    cover_unseen: boolean (optional)
        Whether to black out region of star that is never visibale according to star's inclination.
        Default is True.

    fontsize: int (optional)
        Sets size of axis labels and title (1.5x axis).  Default is 24

    standard_color: boolean (optional)
        If True, code will standardize colormap across all maps.  Default is True 
        If False, no colorbar will be made.

    colorbar: boolean (optional)
        If True, code will generate colorbar for plot of all emaps.  Default is True

    colorbar_label: boolean (optional)
        If True and colorbar is True, will add a label to the colorbar.  Default is True

    indiviudal: boolean (optional)
        If True, will generate individual emap plots along with overall.  Default is True
        Will not generate if no path set to avoid clutter in image display.
        Generates using default parameters from emap_plot function.  
        Plotting parameters passed for overall generally do not affect individual

    standard_indiv: boolean (optional)
        If True, will use standard color map for all individual plots.  Only works if standard_color = True
        Note, if individual = False this will have no effect.

    Returns
    -------
    None
    """    

    
    se("\tPlotting eigenmaps:", dp = dpm)
    se("\t------------------------------------------------", dp = dpm)

    ncurves, ny = eigeny.shape

    se(f"\tSetting projection to {proj}", dp = dpm)

    if proj == "ortho180":
        image = star.map.render(theta=180,projection=proj,rv=False).eval()
    else:
        image = star.map.render(theta=0,projection=proj,rv=False).eval()

    if proj == 'ortho':
        dx = 180.0 / image.shape[1] 
        dy = 180.0 / image.shape[0]

        extent = np.array([
            -90 - dx,  # Left (padded by one pixel width)
            90,       # Right
            -90 - dy,   # Bottom (padded by one pixel height)
            90         # Top
        ])

        fname = 'emaps-ortho.png'

    elif proj == "ortho180":
        dx = 180.0 / image.shape[1] 
        dy = 180.0 / image.shape[0]

        extent = np.array([
            -90 - dx,  # Left (padded by one pixel width)
            90,       # Right
            -90 - dy,   # Bottom (padded by one pixel height)
            90         # Top
        ])

        fname = 'emaps-ortho180.png'

    elif proj == 'rect':
        extent = (-180, 180, -90, 90)
        fname = 'emaps-rect.png'

    elif proj == 'moll':
        dx = 360.0 / image.shape[1] 
        dy = 180.0 / image.shape[0]


        extent = np.array([
            -180 - dx/2,  # Left (padded by one pixel width)
            180 - dx/2,       # Right
            -90 - dy/2,   # Bottom (padded by one pixel height)
            90 - dy/2         # Top
        ])

        fname = 'emaps-moll.png'
        

    ncols = int(np.sqrt(ncurves) // 1)
    nrows = int(ncurves // ncols + (ncurves % ncols != 0))

    if proj == "ortho" or proj == "ortho180":
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False,
                             sharex=True, sharey=True, figsize=(6,7))
    else:
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False,
                             sharex=True, sharey=True, figsize=(10,7))
    
    if individual and emaps_path:
        se(f"\n\tGenerating indivdiual emap plots", dp = dpm)

    rendered_maps = []

    max_fluxes_diff = []

    for k in range(ncurves):
        star.map[:,:] = eigeny[k]

        if proj == "ortho180":
            current_map = star.map.render(theta=180, projection=proj).eval()
        else:
            current_map = star.map.render(theta=0, projection=proj).eval()

        rendered_maps.append(current_map)

        if standard_color:

            amp_array = np.abs(np.array([center_flux - np.nanmin(current_map), np.nanmax(current_map) - center_flux]))

            max_fluxes_diff.append(np.max(amp_array))

    if max_fluxes_diff:
        max_amp = np.max(max_fluxes_diff)

        overall_norm = mpl.colors.Normalize(vmax = max_amp + center_flux, vmin = center_flux - max_amp)
    
    else:
        overall_norm = None
    
    for j in range(ncurves):
        star.map[:,:] = eigeny[j]

        xloc = j %  ncols
        yloc = j // ncols
        ax = axes[yloc, xloc]

        rendered_map = rendered_maps[j]
        
        if standard_color:
            im = ax.imshow(rendered_map,
                    origin="lower",
                    cmap=cmap,
                    extent=extent,
                    norm = overall_norm)
        else:
            im = ax.imshow(rendered_map,
                    origin="lower",
                    cmap=cmap,
                    extent=extent,
                    norm = mpl.colors.CenteredNorm(vcenter=center_flux))

        
        if individual and emaps_path:
            indiv_path = os.path.join(emaps_path, f"{proj}_emaps")

            if not os.path.isdir(indiv_path):
                os.mkdir(indiv_path)

            if standard_indiv:
                emap_plot(star, indiv_path=indiv_path, proj=proj, other_fname=f"emap_{j}_{proj}", cmap = cmap, center_flux=center_flux, 
                    transparent=transparent, cmap_norm=overall_norm)
            else:
                emap_plot(star, indiv_path=indiv_path, proj=proj, other_fname=f"emap_{j}_{proj}", cmap = cmap, center_flux=center_flux, 
                    transparent=transparent, cmap_norm=None)



        if gridlines:

            if not cover_unseen and unseen_line:
                unseen_degree = -1 * star.map.inc.eval()
            else:
                unseen_degree = None

            if proj == "rect":
                ax.grid(color=gridcolor,linestyle=":")
                if not unseen_degree is None:
                    ax.axhline(unseen_degree, c="k",ls="-", lw=3, alpha=1, zorder=1)
                    ax.axhline(unseen_degree, c="deepskyblue",ls="-", lw=2, alpha=1, zorder=1)

            elif proj == "moll":
                lats = lat_lon_lines.get_moll_latitude_lines(dlat=30,unseen_deg=unseen_degree)
                latlines = [None for n in lats]

                for n, l in enumerate(lats):

                    if unseen_degree and n == len(lats) - 1:

                        black_line = ax.plot(
                            l[0], l[1], c="k",ls="-", lw=2, alpha=1, zorder=1
                        )[0]
                        sky_line = ax.plot(
                            l[0], l[1], c="deepskyblue",ls="-", lw=1, alpha=1, zorder=1
                        )[0]
                    else:
                        (latlines[n],) = ax.plot(
                            l[0], l[1], "k:", lw=0.5, alpha=1, zorder=0
                        )

                lons = lat_lon_lines.get_moll_longitude_lines(dlon=60)
                lonlines = [None for n in lons]
                for n, l in enumerate(lons):
                    if (n == 0 or n == len(lons) - 1) and border:
                        (lonlines[n],) = ax.plot(
                            l[0], l[1], "k-", lw=1, alpha=1, zorder=1
                        )
                    else:
                        (lonlines[n],) = ax.plot(
                            l[0], l[1], "k:", lw=0.5, alpha=1, zorder=0
                        )

                # Force the unseen_line to stay strictly inside the outer boundary line
                if unseen_degree and 'sky_line' in locals():

                    left_path = lonlines[0].get_path()
                    right_path = lonlines[-1].get_path()

                    combined_vertices = list(left_path.vertices) + list(right_path.vertices[::-1])
                    closed_oval_path = mpath.Path(combined_vertices)

                    clip_patch = mpatches.PathPatch(closed_oval_path, transform=ax.transData)
        
                    sky_line.set_clip_path(clip_patch)
                    black_line.set_clip_path(clip_patch)

                ax.set_ylim(-93,93)
                ax.set_xlim(-183,183)

            else:
                lats = lat_lon_lines.get_ortho_latitude_lines(unseen_deg=unseen_degree)
                latlines = [None for n in lats]

                for n, l in enumerate(lats):
                    if unseen_degree and n == len(lats) - 1:

                        black_line = ax.plot(
                            l[0], l[1], c="k",ls="-", lw=2, alpha=1, zorder=1
                        )[0]
                        sky_line = ax.plot(
                            l[0], l[1], c="deepskyblue",ls="-", lw=1, alpha=1, zorder=1
                        )[0]

                    else:
                        (latlines[n],) = ax.plot(
                            l[0], l[1], "k:", lw=0.5, alpha=1, zorder=0
                        )

                lons = lat_lon_lines.get_ortho_longitude_lines()
                lonlines = [None for n in lons]
                for n, l in enumerate(lons):
                    if n == 0 or n == (len(lons) - 1) and border:
                        (lonlines[n],) = ax.plot(
                        l[0], l[1], "k-", lw=1.5, alpha=1, zorder=1
                        )
                    else:
                        (lonlines[n],) = ax.plot(
                        l[0], l[1], "k:", lw=.5, alpha=1, zorder=0
                        )
                

                # Force the unseen_line to stay strictly inside the outer boundary line
                if unseen_degree and 'sky_line' in locals():

                    left_path = lonlines[0].get_path()
                    right_path = lonlines[-1].get_path()

                    combined_vertices = list(left_path.vertices) + list(right_path.vertices[::-1])
                    closed_oval_path = mpath.Path(combined_vertices)

                    clip_patch = mpatches.PathPatch(closed_oval_path, transform=ax.transData)
        
                    sky_line.set_clip_path(clip_patch)
                    black_line.set_clip_path(clip_patch)

                plt.xlim(-95,95)
                plt.ylim(-95,95)



        if cover_unseen:
            unseen_degree = -1 * star.map.inc.eval()

            if proj == "rect":

                ax.fill_between(np.linspace(-180,180,361), -90, unseen_degree, color="k", alpha = .6, edgecolor='none')

                pass

            elif proj == "moll":

                lats = lat_lon_lines.get_moll_latitude_lines(dlat=0, unseen_deg=unseen_degree)

                x_vals = np.array(lats[0][0])
                fill_val = np.array(lats[0][1])

                lat_xs = x_vals[~np.isnan(fill_val)]  

                lons = lat_lon_lines.get_moll_longitude_lines(dlon=360)
                lonlines = [None for n in lons]

                lon_xs = np.array([])
                lon_ys = np.array([])

                for n, l in enumerate(lons):
                    x_is = np.argsort(l[0])

                    lon_xs = np.append(lon_xs,l[0][x_is])
                    lon_ys = np.append(lon_ys,l[1][x_is])

                x_mask = (lon_xs > np.min(lat_xs)) & (lon_xs < np.max(lat_xs))

                y_mask = (lon_ys < np.nanmedian(fill_val))

                xs = lon_xs[x_mask & y_mask]

                ys = lon_ys[x_mask & y_mask]

                ax.fill_between(xs, ys, np.nanmedian(fill_val), 
                        color="k", alpha = .6, edgecolor='none')

            else:
                lats = lat_lon_lines.get_ortho_latitude_lines(dlat=0, unseen_deg=unseen_degree)

                x_vals = np.array(lats[0][0])
                fill_val = np.array(lats[0][1])

                lat_xs = x_vals[~np.isnan(fill_val)]  

                lons = lat_lon_lines.get_ortho_longitude_lines(dlon=180)
                lonlines = [None for n in lons]

                lon_xs = np.array([])
                lon_ys = np.array([])

                for n, l in enumerate(lons):
                    x_is = np.argsort(l[0])

                    lon_xs = np.append(lon_xs,l[0][x_is])
                    lon_ys = np.append(lon_ys,l[1][x_is])

                x_mask = (lon_xs > np.min(lat_xs)) & (lon_xs < np.max(lat_xs))

                y_mask = (lon_ys < np.nanmedian(fill_val))

                xs = lon_xs[x_mask & y_mask]

                ys = lon_ys[x_mask & y_mask]

                ax.fill_between(xs, ys, np.nanmedian(fill_val), 
                        color="k", alpha = .6, edgecolor='none')
        

        if not ticks or proj != "rect":
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.tick_params(left=False, bottom=False)
        else:
            ax.set_xticks([-90,0,90])
            ax.set_xticklabels([-90,0,90])

            ax.set_yticks([-45,0,45])
            ax.set_yticklabels([-45,0,45])

            ax.tick_params(direction="in")
        
        if not border or proj != "rect":
            ax.set_frame_on(False)
          

    if labels:
        fig.supxlabel("Longitude [deg]",fontsize=fontsize)
        fig.supylabel("Latitude [deg]",fontsize=fontsize)

    if title:
        fig.suptitle(title,fontsize=int(fontsize*1.5))

    fig.tight_layout()

    if colorbar and standard_color:
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.95, aspect = 40, pad = .03)
        if colorbar_label:
            cbar.set_label("Flux [Normalized]", size=12, rotation = 270, labelpad = 15)

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


def create_eflux(star, eigeny, emaps_path=None, theta = np.linspace(-180, 180, 361),
                 transparent=False, fontsize=16, centerline=True,
                 labels=True, title = None, border=True, ticks=False, cline_color="k",
                 individual=True, color="sandybrown"):

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

    gridlines: boolean (optional)
        If True, will add grid lines to overall plot.  Default is False

    labels: boolean (optional)
        Wether to add axis labels to overall plot (a represents all).  Default to False

    title: str (optional)
        If value passed, will set it as the title.  Default is None

    border: boolean (optional)
        Wether to add border around each axis plot on overall plot.  Default to False

    ticks: boolean (optional)
        If True, will add ticks and values to overall plot.  Default is False

    gridcolor: string (optional)
        Color to set curve to be (see https://matplotlib.org/stable/gallery/color/named_colors.html).  
        Default is 'darkgrey'.

    indiviudal: boolean (optional)
        If True, will generate individual emap plots along with overall.  Default is True
        Will not generate if no path set to avoid clutter in image display.

    color: string (optional)
        Color to set curve to be (see https://matplotlib.org/stable/gallery/color/named_colors.html).  
        Default is 'sandybrown'.

    #Leaving this here because I like darkcyan with the plasma cmap (starry default)

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
    
    for j in range(ncurves):
        star.map[:,:] = eigeny[j]

        xloc = j %  ncols
        yloc = j // ncols
        ax = axes[yloc, xloc]

        j_flux = star.map.flux(theta=theta).eval()
        
        ax.plot(theta, j_flux,color=color)

        if individual and emaps_path:
            indiv_path = os.path.join(emaps_path, f"rlc")

            if not os.path.isdir(indiv_path):
                os.mkdir(indiv_path)
            
            create_rv.flux_rv_line(star,theta,flux=j_flux,flux_name=f"{indiv_path}/rlc_{j}",flux_only=True,
                                   color=color)

        if not ticks:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.tick_params(left=False, bottom=False)
        else:
            ax.set_xticks([-90,0,90])
            ax.set_xticklabels([-90,0,90])

            ax.set_xlim(-180,180)

            ax.tick_params(direction="in")

            min_f = j_flux.min()
            max_f = j_flux.max()
            if max_f - min_f < 1e-10:
                interval = max_f * .05
                y_tick_values = [max_f] 
                ax.set_yticks(y_tick_values)
                ax.set_yticklabels([f"{val:.2f}" for val in y_tick_values])
                ax.set_ylim(min_f - interval * 1.05, max_f + interval * 1.05)

            else:
                amp = max_f - min_f

                buffer = 0.05 

                middle = (max_f + min_f) / 2
                y_tick_values = [min_f, middle, max_f] 
                ax.set_yticks(y_tick_values)
                ax.set_yticklabels([f"{val:.2f}" for val in y_tick_values])
                ax.set_ylim(min_f - amp*buffer, max_f + amp*buffer)

        
        if not border:
            ax.set_frame_on(False)

        if centerline:
            middle = (np.max(j_flux) + np.min(j_flux)) / 2
            ax.axhline(y=middle,color=cline_color,linestyle=":")

    se(f"\tGenerating overall light curve plot", dp = dpm)

    if labels:
        fig.supxlabel("Angle of rotation [degrees]",fontsize=fontsize)
        fig.supylabel("Flux [normalized]",fontsize=fontsize)

    if title:
        fig.suptitle(title,fontsize=int(fontsize*1.5))
    
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
        se('\033[31mERROR:\033[0m' + ' Call structure is "\033[34mpython create_emaps.py <configuration file>\033[0m"',dp = dpm)
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

