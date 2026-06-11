"""
[Description needed here]
https://github.com/pb-aj/un-spot-able
"""

#general imports
import os
import sys
import numpy as np
import matplotlib
import matplotlib as mpl
matplotlib.rcParams['axes.formatter.useoffset'] = False
matplotlib.use('TkAgg') #test if this line is needed
import matplotlib.pyplot as plt
import starry2 as starry
from cmcrameri import cm
import faulthandler
faulthandler.enable()

# *NOTE* To have transparent gifs, need to install ImageMagick and set path to it here.
# To find the correct path, can use which magick in terminal after installing
try:
    plt.rcParams['animation.convert_path'] = '/opt/homebrew/bin/magick'
    pass
except:
    pass

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


def flux_rv_line(rv_star,theta = np.linspace(-180, 180, 361), flux=None, rv=None,
                 flux_name=None,rv_name=None,flux_only=False,rv_only=False,
                 transparent=False, labels=True, title = None, border=True, ticks=True, 
                 centerline=True, fontsize=16, color="sandybrown", cline_color="k"):
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
        if flux is None:
            flux = rv_star.map.flux(theta=theta).eval()
        
        # Plot the flux
        plt.figure(figsize=(12, 5))

        plt.plot(theta, flux, color=color)


        if labels:
            plt.xlabel("Angle of rotation [degrees]", fontsize=fontsize)
            plt.ylabel("Flux [normalized]", fontsize=fontsize)

        if title:
            plt.title(title,fontsize=fontsize*1.5)

        if centerline:
            middle = (np.max(flux) + np.min(flux)) / 2
            plt.axhline(y=middle,color=cline_color,linestyle=":")

        if not ticks:
            plt.gca().set_xticklabels([])
            plt.gca().set_yticklabels([])
            plt.tick_params(left=False, bottom=False)
        else:
            plt.gca().set_xticks([-135,-90,-45,0,45,90,135])
            plt.gca().set_xticklabels([-135,-90,-45,0,45,90,135])

            plt.xlim(-180,180)

            min_f = np.min(flux)
            max_f = np.max(flux)

            plt.tick_params(direction="in")

            if max_f - min_f < 1e-10:
                interval = max_f * .05
                y_tick_values = [max_f - interval, max_f, max_f + interval] 
                plt.yticks(y_tick_values)
                plt.gca().set_yticklabels([f"{val:.2f}" for val in y_tick_values])
                plt.ylim(min_f - interval * 1.05, max_f + interval * 1.05)

            else:
                amp = max_f - min_f

                buffer = 0.05 

                middle = (max_f + min_f) / 2
                y_tick_values = [min_f, (min_f + middle) / 2, middle, (max_f + middle) / 2, max_f] 
                plt.yticks(y_tick_values)
                plt.gca().set_yticklabels([f"{val:.2f}" for val in y_tick_values])
                plt.ylim(min_f - amp*buffer, max_f + amp*buffer)

        if not border:
            plt.gca().set_frame_on(False)

        plt.tight_layout()
        if flux_name:
            plt.savefig(flux_name, dpi = 300, transparent=transparent)
        else:
            plt.show()
        plt.close()

    if not flux_only:
        if rv is None:
            rv = rv_star.map.rv(theta=theta).eval()

        # Plot the radial velocity
        plt.figure(figsize=(12, 5))
        plt.plot(theta, rv, color=color)

        if labels:
            plt.xlabel("Angle of rotation [degrees]", fontsize=fontsize)
            plt.ylabel("Radial Velocity [m/s]", fontsize=fontsize)

        if title:
            plt.title(title,fontsize=fontsize*1.5)

        if centerline:
            plt.axhline(y=0,color=cline_color,linestyle=":")

        if not ticks:
            plt.gca().set_xticklabels([])
            plt.gca().set_yticklabels([])
            plt.tick_params(left=False, bottom=False)

        else:
            plt.gca().set_xticks([-135,-90,-45,0,45,90,135])
            plt.gca().set_xticklabels([-135,-90,-45,0,45,90,135])

            plt.xlim(-180,180)

            min_f = np.min(rv)
            max_f = np.max(rv)

            plt.tick_params(direction="in")

            amp = max_f - min_f

            buffer = 0.05 

            y_tick_values = [min_f, (min_f) / 2, 0, (max_f) / 2, max_f] 
            plt.yticks(y_tick_values)
            plt.gca().set_yticklabels([f"{val:.2f}" for val in y_tick_values])
            plt.ylim(min_f - amp*buffer, max_f + amp*buffer)


        if not border:
            plt.gca().set_frame_on(False)
        
        plt.tight_layout()
        if rv_name:
            plt.savefig(rv_name, dpi = 300, transparent=transparent)
        else:
            plt.show()
        plt.close()



def multi_phase_plot(rv_star,fname=None,cmap=cm.bam,center_flux=0):
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

        rv_star.map.show(rv=False,theta=degree,ax=ax,figsize=(5,5),cmap=cmap)
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

def map_animations(rv_star,theta = np.linspace(0, 360, 181)[:-1], fname=None, cmap = cm.buda,
                   maps_only = False, flux_only = False, rv_only = False, color="sandybrown",
                   interval = 75, fps = 10, fontsize=16, map_gridlines=True, map_labels=True, norm=None,
                   colorbar="bottom", colorbar_label=True, transparent=False, marker_color = "darkgrey",
                   curve_border=True, curve_labels=True, ticks=True, legend=False, curve_gridlines=False,
                   centerline=True, cline_color="k",guideline=True, guideline_color="k", 
                   dpi=300, html5_video=True):
    """
    {desc} - needs so muuch work vvvv

    Arguments
    ---------
    rv_star: object
        A starry star object (w/rv=True), initialized with a cfg file

    theta: 2D array (optional)
        Degrees to evaluate light curve at for each emap.  Default is every degree - np.linspace(0, 360, 360)

    fname: string (optional)
        Name of file to save animation to (must end in .mp4 or .gif).  Default is None
        If None, will display image instead of saving
        For .gif(), it is recommended to reduce size of theta

    interval: int (optional)
        Delay between video frames in milliseconds (see matplotlib.animation.FuncAnimation)
        Default is 75

    fps: int (optional)
        Frames per second to run gif at. Default is 50
        *NOTE* 50 fps is the maximum allowed for Pillow gif's (did not test other packages)

    fontsize: int (optional)
        Sets size of axis labels and title (1.5x axis).  Default is 24

    map_gridlines: boolean (optional)
        If True, will add grid lines to ortho map.  Default is True

    map_border: boolean (optional)
        Wether to add border around flux/RV maps.  Default to False

    map_labels: boolean (optional)
        Wether to add axis and title labels to flux/RV maps.  Default to True

    colorbar: boolean or "bottom" (optional)
        Whether to include a color bar or not.  Default is False
        When True colorbar is on right
        Using "bottom" places the colorbar on the bottom 

    marker_color: string (optional)
        Color to set curve to be (see https://matplotlib.org/stable/gallery/color/named_colors.html).  
        Default is 'dimgrey'.

    transparent: boolean (optional)
        Whether to make plots transparent.  Default to False
        *NOTE* This will only work if passed filename ends in .gif (transparent not easily supported by .mp4)

    *NOTE* The following only apply if an RV/Flux curve is requested
    as applying them to the ortho projection produces incorrect results

    curve_border: boolean (optional)
        Wether to add border around each axis plot.  Default to True

    curve_labels: boolean (optional)
        Wether to add axis and title labels to flux/RV curves.  Default to True

    ticks: boolean (optional)
        If True, will add ticks and values to flux/RV curves.  Default is False

    legend: boolean (optional)
        If True, will include legend on flux/RV curves showing current value.  Default is True

    curve_gridlines: boolean (optional)
        If True, will add grid lines to flux/RV curves.  Default is False

    Returns
    -------
    None
    """

    if rv_only or flux_only:

        if not rv_only:

            if maps_only:
                fig, axes = plt.subplots(nrows=1, ncols=1, squeeze=False,
                                    sharex=False, sharey=False, figsize=(7, 8))
                
                if map_labels:
                    axes[0,0].set_title("Flux Projection", fontsize=fontsize)

                plt.tight_layout()

                if colorbar_label and colorbar:
                    plt.subplots_adjust(hspace=0.25)
                    flux_cbar_label = "Flux [Normalized]"

                else:
                    flux_cbar_label = None
            
                if norm is None:
                    rv_star.map.show(theta=theta, ax=axes[0,0], rv=False,show_image=True, colorbar=colorbar, 
                                    colorbar_label = flux_cbar_label,
                                    cmap=cmap, grid=map_gridlines, colorbar_size="2.5%",
                                    file=fname, dpi = dpi, html5_video=html5_video,
                                    transparent=transparent, interval=interval, fps=fps)
                else:
                    rv_star.map.show(theta=theta, ax=axes[0,0], rv=False,show_image=True, colorbar=colorbar, 
                                    norm=norm, colorbar_label = flux_cbar_label, cmap=cmap,
                                    grid=map_gridlines, colorbar_size="2.5%",
                                    file=fname, dpi = dpi, html5_video=html5_video,
                                    transparent=transparent, interval=interval, fps=fps)
            
            else:

                fig, axes = plt.subplots(nrows=2, ncols=1, squeeze=False,
                                sharex=False, sharey=False, figsize=(7, 6))
            
                ax_flux = axes[1,0]

                new_theta = np.linspace(-180, 180, theta.shape[0]+1)[:-1]
                flux_data = rv_star.map.flux(theta=new_theta).eval()

                flux_image = ax_flux.plot(new_theta, flux_data,label="sample",color=color)                    

                if centerline:
                    middle = (np.max(flux_data) + np.min(flux_data)) / 2
                    ax_flux.axhline(y=middle,color=cline_color,linestyle="--",linewidth=.5)

                if guideline:
                    ax_flux.axvline(x=0, color=guideline_color, linestyle=':', linewidth=1.5)  


                if curve_labels:
                    ax_flux.set_xlabel("Angle of rotation [degrees]", fontsize=fontsize)
                    ax_flux.set_ylabel("Flux [normalized]", fontsize=fontsize)

                if map_labels:
                    axes[0,0].set_title("Flux Projection", fontsize=fontsize)

                if legend:
                    L_flux = ax_flux.legend(loc="upper left",handlelength=0)

                if curve_gridlines:
                    ax_flux.grid(color=marker_color,linestyle=":")

                if not ticks:
                    ax_flux.set_xticklabels([])
                    ax_flux.set_yticklabels([])
                    ax_flux.tick_params(left=False, bottom=False)


                if not curve_border:
                    ax_flux.set_frame_on(False)

                plt.tight_layout()

                if colorbar_label and colorbar:
                    plt.subplots_adjust(hspace=0.25)
                    flux_cbar_label = "Flux [Normalized]"

                else:
                    flux_cbar_label = None

                if not curve_border:
                    ax_flux.set_frame_on(False)

                if legend:
                    legend_list = [L_flux]
                else:
                    legend_list = None

                if norm is None:
                    rv_star.map.show(theta=theta,rv=False, ax=axes[0,0], 
                                        colorbar_label=flux_cbar_label, show_image=True,
                                        colorbar=colorbar, grid=map_gridlines,
                                        cmap=cmap,
                                        extra_lines = [(flux_data,flux_image)],
                                        legend_list = legend_list,
                                        file=fname, dpi = dpi, html5_video=html5_video,
                                        transparent=transparent, interval=interval, fps=fps)
                else:
                    rv_star.map.show(theta=theta,rv=False, ax=axes[0,0], 
                                        colorbar_label=flux_cbar_label, show_image=True,
                                        colorbar=colorbar, grid=map_gridlines,
                                        cmap=cmap, norm=norm,
                                        extra_lines = [(flux_data,flux_image)],
                                        legend_list = legend_list,
                                        file=fname, dpi = dpi, html5_video=html5_video,
                                        transparent=transparent, interval=interval, fps=fps)

        if not flux_only:    
            if maps_only:
                fig, axes = plt.subplots(nrows=1, ncols=1, squeeze=False,
                                    sharex=False, sharey=False, figsize=(7, 8))
                
                if map_labels:
                    axes[0,0].set_title("Radial Velocity Projection", fontsize=fontsize)

                plt.tight_layout()

                if colorbar_label and colorbar:
                    plt.subplots_adjust(hspace=0.25)
                    rv_cbar_label = "Line of Sight Velocity [m/s]"

                else:
                    rv_cbar_label = None

                rv_star.map.show(theta=theta,rv=True, ax=axes[0,0],show_image=True, colorbar=colorbar, 
                            norm=matplotlib.colors.CenteredNorm(), colorbar_label = rv_cbar_label,
                            grid=map_gridlines, colorbar_size="2.5%",
                            file=fname, dpi = dpi, html5_video=html5_video,
                            transparent=transparent, interval=interval, fps=fps)
            
            else:
                fig, axes = plt.subplots(nrows=2, ncols=1, squeeze=False,
                                sharex=False, sharey=False, figsize=(7, 6))
            
                ax_rv = axes[1,0]

                new_theta = np.linspace(-180, 180, theta.shape[0]+1)[:-1]

                rv_data = rv_star.map.rv(theta=new_theta).eval()       

                rv_image = ax_rv.plot(new_theta, rv_data,label="sample",color=color)

                if centerline:
                    ax_rv.axhline(y=0,color=cline_color,linestyle="--",linewidth=.5)

                if guideline:
                    ax_rv.axvline(x=0, color=guideline_color, linestyle=':', linewidth=1.5)


                if curve_labels:
                    ax_rv.set_xlabel("Angle of rotation [degrees]", fontsize=fontsize)
                    ax_rv.set_ylabel("Radial velocity [m/s]", fontsize=fontsize)

                if map_labels:
                    axes[0,0].set_title("Radial Velocity Projection", fontsize=fontsize)
                

                if legend:
                    L_rv = ax_rv.legend(loc="upper left",handlelength=0)

                if curve_gridlines:
                    ax_rv.grid(color=marker_color,linestyle=":")

                if not ticks:
                    ax_rv.set_xticklabels([])
                    ax_rv.set_yticklabels([])
                    ax_rv.tick_params(left=False, bottom=False)

                if not curve_border:
                    ax_rv.set_frame_on(False)

                plt.tight_layout()

                if colorbar_label and colorbar:
                    plt.subplots_adjust(hspace=0.25)
                    rv_cbar_label = "Line of Sight Velocity [m/s]"

                else:
                    rv_cbar_label = None

                if not curve_border:
                    ax_rv.set_frame_on(False)
                
                if legend:
                    rv_star.map.show(theta=theta,rv=True, ax=axes[0,0],show_image=True, colorbar=colorbar, 
                                    norm=matplotlib.colors.CenteredNorm(), colorbar_label= rv_cbar_label,
                                    grid=map_gridlines,
                                    extra_lines = [(rv_data,rv_image)],
                                    legend_list = [L_rv],
                                    file=fname, dpi = dpi, html5_video=html5_video,
                                    transparent=transparent, interval=interval, fps=fps)
                else:
                    rv_star.map.show(theta=theta,rv=True, ax=axes[0,0],show_image=True, colorbar=colorbar, 
                                    norm=matplotlib.colors.CenteredNorm(), colorbar_label = rv_cbar_label,
                                    grid=map_gridlines,
                                    extra_lines = [(rv_data,rv_image)],
                                    file=fname, dpi = dpi, html5_video=html5_video,
                                    transparent=transparent, interval=interval, fps=fps)
    else:

        if maps_only:
            fig, axes = plt.subplots(nrows=1, ncols=2, squeeze=False,
                                sharex=False, sharey=False, figsize=(10, 6))
            
            if map_labels:
                axes[0,0].set_title("Flux Projection", fontsize=fontsize)
                axes[0,1].set_title("Radial Velocity Projection", fontsize=fontsize)

            plt.tight_layout()

            if colorbar_label and colorbar:
                plt.subplots_adjust(hspace=0.25)
                flux_cbar_label = "Flux [Normalized]"
                rv_cbar_label = "Line of Sight Velocity [m/s]"

            else:
                flux_cbar_label = None
                rv_cbar_label = None

            if norm is None:
                img1,image1,lonlines1,latlines1 = rv_star.map.show(theta=theta,rv=False, ax=axes[0,0], 
                                                                   colorbar_label=flux_cbar_label, show_image=False,
                                                                   colorbar=colorbar, grid=map_gridlines,
                                                                   cmap=cmap, colorbar_size="2.5%", 
                                                                   )
            else:
                img1,image1,lonlines1,latlines1 = rv_star.map.show(theta=theta,rv=False, ax=axes[0,0], 
                                                                   colorbar_label=flux_cbar_label, show_image=False,
                                                                   colorbar=colorbar, grid=map_gridlines, colorbar_size="2.5%", 
                                                                   cmap=cmap, norm=norm)
        

            rv_star.map.show(theta=theta,rv=True, ax=axes[0,1],show_image=True, colorbar=colorbar, 
                            norm=matplotlib.colors.CenteredNorm(), colorbar_label = rv_cbar_label,
                            grid=map_gridlines, colorbar_size="2.5%",
                            extra_image=[img1,image1,lonlines1,latlines1],
                            file=fname, dpi = dpi, html5_video=html5_video,
                            transparent=transparent, interval=interval, fps=fps)
            
            
        else:
            fig, axes = plt.subplots(nrows=2, ncols=2, squeeze=False,
                                sharex=False, sharey=False, figsize=(12, 6))
            
            ax_flux = axes[1,0]
            ax_rv = axes[1,1]

            new_theta = np.linspace(-180, 180, theta.shape[0]+1)[:-1]
            flux_data = rv_star.map.flux(theta=new_theta).eval()

            rv_data = rv_star.map.rv(theta=new_theta).eval()

            flux_image = ax_flux.plot(new_theta, flux_data,label="sample",color=color)                    

            rv_image = ax_rv.plot(new_theta, rv_data,label="sample",color=color)

            if centerline:
                middle = (np.max(flux_data) + np.min(flux_data)) / 2
                ax_flux.axhline(y=middle,color=cline_color,linestyle="--",linewidth=.5)

                ax_rv.axhline(y=0,color=cline_color,linestyle="--",linewidth=.5)

            if guideline:
                ax_flux.axvline(x=0, color=guideline_color, linestyle=':', linewidth=1.5)  
                ax_rv.axvline(x=0, color=guideline_color, linestyle=':', linewidth=1.5)


            if curve_labels:
                ax_flux.set_xlabel("Angle of rotation [degrees]", fontsize=fontsize)
                ax_flux.set_ylabel("Flux [normalized]", fontsize=fontsize)

                ax_rv.set_xlabel("Angle of rotation [degrees]", fontsize=fontsize)
                ax_rv.set_ylabel("Radial velocity [m/s]", fontsize=fontsize)

            if map_labels:
                axes[0,0].set_title("Flux Projection", fontsize=fontsize)
                axes[0,1].set_title("Radial Velocity Projection", fontsize=fontsize)
            

            if legend:
                L_flux = ax_flux.legend(loc="upper left",handlelength=0)
                L_rv = ax_rv.legend(loc="upper left",handlelength=0)

            if curve_gridlines:
                ax_flux.grid(color=marker_color,linestyle=":")
                ax_rv.grid(color=marker_color,linestyle=":")

            if not ticks:
                ax_flux.set_xticklabels([])
                ax_flux.set_yticklabels([])
                ax_flux.tick_params(left=False, bottom=False)

                ax_rv.set_xticklabels([])
                ax_rv.set_yticklabels([])
                ax_rv.tick_params(left=False, bottom=False)

            if not curve_border:
                ax_flux.set_frame_on(False)

                ax_rv.set_frame_on(False)

            plt.tight_layout()

            if colorbar_label and colorbar:
                plt.subplots_adjust(hspace=0.25)
                flux_cbar_label = "Flux [Normalized]"
                rv_cbar_label = "Line of Sight Velocity [m/s]"

            else:
                flux_cbar_label = None
                rv_cbar_label = None

            if not norm is None:
                img1,image1,lonlines1,latlines1 = rv_star.map.show(theta=theta,rv=False, ax=axes[0,0], colorbar_label=flux_cbar_label,
                                                                show_image=False,colorbar=colorbar, grid=map_gridlines,
                                                                cmap=cmap, norm=norm)
            else:
                img1,image1,lonlines1,latlines1 = rv_star.map.show(theta=theta,rv=False, ax=axes[0,0], colorbar_label=flux_cbar_label,
                                                                show_image=False,colorbar=colorbar, grid=map_gridlines,
                                                                cmap=cmap)

            if not curve_border:
                ax_flux.set_frame_on(False)
                ax_rv.set_frame_on(False)
            
            if legend:
                rv_star.map.show(theta=theta,rv=True, ax=axes[0,1],show_image=True, colorbar=colorbar, 
                                norm=matplotlib.colors.CenteredNorm(), colorbar_label= rv_cbar_label,
                                grid=map_gridlines,
                                extra_image=[img1,image1,lonlines1,latlines1],
                                extra_lines = [(flux_data,flux_image),(rv_data,rv_image)],
                                legend_list = [L_flux,L_rv],
                                file=fname, dpi = dpi, html5_video=html5_video,
                                transparent=transparent, interval=interval, fps=fps)
            else:
                rv_star.map.show(theta=theta,rv=True, ax=axes[0,1],show_image=True, colorbar=colorbar, 
                                norm=matplotlib.colors.CenteredNorm(), colorbar_label = rv_cbar_label,
                                grid=map_gridlines,
                                extra_image=[img1,image1,lonlines1,latlines1],
                                extra_lines = [(flux_data,flux_image),(rv_data,rv_image)],
                                file=fname, dpi = dpi, html5_video=html5_video,
                                transparent=transparent, interval=interval, fps=fps)
    plt.close(fig)



def create_rv(eigeny, fit, rv_path, theta = np.linspace(0, 360, 181)):
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

    lmax = cfg.sim.lmax
    udeg = cfg.star.udeg
    rv_star = utils.initstar(fit, lmax, udeg=udeg, include_rv=True)
    uni_star = utils.initstar(fit, lmax, udeg=udeg, include_rv=True)
    ncurves, ny = eigeny.shape

    ani_interval = int(6000 / len(theta))

    se("\tLooping through each map and generating plots:", dp=dpm)
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

        uni_star.map[0,0] = rv_star.map[0,0]
        image = uni_star.map.render(theta=180,projection="rect",rv=False).eval()
        
        map_animations(rv_star, theta = theta[:-1],fname=f"{indiv_path}/emap_animation.mp4", 
                       interval = ani_interval, transparent=False)
        
        map_animations(rv_star, theta = theta[::2][:-1],fname=f"{indiv_path}/emap_animation.gif", 
                       interval = ani_interval, transparent=False)
        
        create_emaps.emap_plot(rv_star, indiv_path=indiv_path, proj='rect', other_fname=None, 
                 transparent=False, colorbar=True, center_flux=np.nanmean(image))
        
        create_emaps.emap_plot(rv_star, indiv_path=indiv_path, proj='moll', other_fname=None, 
                 transparent=False, colorbar=True, center_flux=np.nanmean(image))

        flux_rv_line(rv_star,
                     flux_name=f"{rv_path}/map_{i}/flux_curve.png",rv_name=f"{rv_path}/map_{i}/rv_curve.png")
        
        # multi_phase_plot(rv_star,fname=f"{rv_path}/map_{i}/map_slideshow.png",center_flux=np.nanmean(image))    

        se(f'\033[38;5;208m\t\t"Emap {i}" plots are complete!\033[0m', dp=dpm)
        se("\t\t------------------------------------------------", dp=dpm)

    se(f"\tAll plots have been saved to:\n\t\033[34m{rv_path}\033[0m", dp = dpm)
    se("----------------------------------------------------------------------------",dp = dpm)








if __name__ == "__main__":
    # Uncomment if you want to see command line arguments 
    # se(sys.argv,dp = dpm) 

    # Check command line input is correct
    if len(sys.argv) < 2:
        se("\n----------------------------------------------------------------------------",dp = dpm)
        se('\033[31mERROR:\033[0m' + ' Call structure is "\033[34mpython create_rv.py <configuration file>\033[0m"',dp = dpm)
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

    create_rv(eigeny, fit, rv_path)

    sys.exit("\033[32mdone\033[0m")