import os
import numpy as np
import matplotlib as mpl
mpl.rcParams['axes.formatter.useoffset'] = False
import matplotlib.pyplot as plt


def emaps(star, eigeny, outdir=None, proj='ortho', transparent=False, plot_name=None):
    ncurves, ny = eigeny.shape
    eigenmaps = []

    if proj == 'ortho':
        extent = (-90, 90, -90, 90)
        fname = 'emaps-ecl.png'
    elif proj == 'rect':
        extent = (-180, 180, -90, 90)
        fname = 'emaps-rect.png'
    elif proj == 'moll':
        extent = (-180, 180, -90, 90)
        fname = 'emaps-moll.png'

    if plot_name:
        fname = plot_name

    lmax = np.int(ny**0.5 - 1)

    ncols = np.int(np.sqrt(ncurves) // 1)
    nrows = np.int(ncurves // ncols + (ncurves % ncols != 0))
    npane = ncols * nrows

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False,
                             sharex=True, sharey=True)
    
    for j in range(ncurves):
        star.map[1:,:] = 0

        xloc = j %  ncols
        yloc = j // ncols
        ax = axes[yloc, xloc]
        
        yi = 1
        for l in range(1, lmax + 1):
            for m in range(-l, l + 1):
                star.map[l, m] = eigeny[j, yi]
                yi += 1
        
        ax.imshow(star.map.render(theta=0, projection=proj).eval(),
                  origin="lower",
                  cmap="plasma",
                  extent=extent)
    
    
        eigenmaps.append(star.map.render(theta=0, projection=proj).eval())

        # Axes are wrong for non-rectangular projections
        if proj == 'ortho' or proj == 'moll':
            ax.axis('off')

    # Empty subplots
    for j in range(ncurves, npane):
        xloc = j %  ncols
        yloc = j // ncols
        ax = axes[yloc, xloc]

        ax.axis('off')

    fig.tight_layout()
    
    if outdir:
        plt.savefig(os.path.join(outdir, fname), transparent=transparent)
    else:
        plt.show()
    plt.close(fig)

    return np.array(eigenmaps)

"""
Function added by A.J. deVaux to plot flux of each rotating eigenmap
"""
def eflux(star, eigeny, outdir, transparent=False):

    ncurves, ny = eigeny.shape

    lmax = int(ny**0.5 - 1)
    thet = np.linspace(0, 360, ny)

    fig, ax = plt.subplots(nrows=ncurves, ncols=1, figsize=(10, ncurves))

    for j in range(ncurves):
        star.map[1:,:] = 0
        yi = 1
        for l in range(1, lmax + 1):
            for mn in range(-l, l + 1):
                star.map[l, mn] = eigeny[j, yi]
                yi += 1
        
        ax[j].plot(star.map.flux(theta=thet).eval())
    
    fig.tight_layout()
    plt.savefig(f"{outdir}/light_curve", transparent=transparent)
    plt.close(fig)

"""
Function added by A.J. deVaux to plot flux of each rotating eigenmap
"""
def erv_curves(star, eigeny, outdir, transparent=False):

    ncurves, ny = eigeny.shape
    lmax = int(ny**0.5 - 1)
    theta = np.linspace(0, 360, ny)

    fig, ax = plt.subplots(nrows=ncurves, ncols=1, figsize=(20, ncurves*3))

    for j in range(ncurves):
        star.map[1:,:] = 0
        yi = 1
        for l in range(1, lmax + 1):
            for mn in range(-l, l + 1):
                star.map[l, mn] = eigeny[j, yi]
                yi += 1

        ax[j].plot(theta, star.map.rv(theta=theta).eval())
        ax[j].set_xlabel("Angle of rotation [degrees]")
        ax[j].set_ylabel("Radial velocity [m/s]")

        

    fig.tight_layout()
    plt.savefig(f"{outdir}/rv_curve", transparent=transparent)
    plt.close(fig)

    pass





def lightcurves(t, lcs, outdir):
    nharm, nt = lcs.shape
    
    l =  1
    m = -1
    pos = True

    fig, ax = plt.subplots(1, figsize=(8,5))
    
    for i in range(nharm):
        plt.plot(t, lcs[i], label=r"${}Y_{{{}{}}}$".format(["-", "+"][pos],
                                                           l, m))
        if pos:
            pos = False
        else:
            pos = True
            if l == m:
                l += 1
                m  = -l               
            else:
                m += 1
            
    plt.ylabel('Normalized Flux')
    plt.xlabel('Time (days)')
    plt.legend(ncol=l, fontsize=6)
    fig.tight_layout()
    plt.savefig(os.path.join(outdir, 'lightcurves.png'))
    plt.close(fig)

def eigencurves(t, lcs, outdir, ncurves=None):
    if type(ncurves) == type(None):
        ncurves = lcs.shape[0]

    fig, ax = plt.subplots(1, figsize=(8,5))    

    for i in range(ncurves):
        plt.plot(t, lcs[i], label="E-curve {}".format(i+1))

    plt.ylabel('Normalized Flux')
    plt.xlabel('Time (days)')

    plt.legend(fontsize=6)
    fig.tight_layout()
    plt.savefig(os.path.join(outdir, 'eigencurves.png'))
    plt.close(fig)
    
def ecurvepower(evalues, outdir):
    ncurves = len(evalues)
    num = np.arange(1, ncurves + 1)

    fig, axes = plt.subplots(nrows=2)
    
    axes[0].plot(num, evalues / np.sum(evalues), 'ob')
    axes[0].set_xlabel('E-curve Number')
    axes[0].set_ylabel('Normalized Power')

    axes[1].semilogy(num, evalues / np.sum(evalues), 'ob')
    axes[1].set_xlabel('E-curve Number')
    axes[1].set_ylabel('Normalized Power')

    fig.tight_layout()
    plt.savefig(os.path.join(outdir, 'ecurvepower.png'))
    plt.close(fig)
