#general imports
import os
import sys
import numpy as np
import matplotlib
matplotlib.rcParams['axes.formatter.useoffset'] = False
import matplotlib.pyplot as plt
import starry2 as starry
import faulthandler
faulthandler.enable()


# Directory structure
maindir    = os.path.dirname(os.path.realpath(__file__))
libdir     = os.path.join(maindir, 'lib')

# Lib imports
sys.path.append(libdir)
print(sys.path)
from lib import utils
from lib import fitclass    as fc

# py imports
import create_eigens
import create_rv

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


def plot_intensity(realistic_star, theta, fname):
    fig, axes = plt.subplots(nrows=2, ncols=1, squeeze=False,
                            sharex=False, sharey=False, figsize=(12, 6))

    ax = axes[1,0]

    new_theta = np.linspace(-180, 180, 61)[:-1]

    limb = realistic_star.map.render(rv=False,theta=new_theta).eval()[:,150,150]

    limb_image = ax.plot(new_theta, limb,label="sample")
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1)
    ax.axhline(y=uni_value*1.35, color='red', linestyle='--', linewidth=1,label="sample")
    L_limb = ax.legend(loc="upper left",handlelength=0)

    ax.set_xlabel("Angle of rotation [degrees]", fontsize=16)
    ax.set_ylabel("Center Intensity", fontsize=16)

    ax.set_title("Intensity at Central Point (lon/lat = 0)",fontsize=20)

    plt.tight_layout()


    realistic_star.map.show(theta=theta,rv=False, ax=axes[0,0],show_image=True, colorbar="bottom", 
                            extra_lines = [(limb,limb_image)],
                            legend_list = [L_limb],
                            uni_int = uni_value,
                            file=fname,
                            dpi = 300)
    
    plt.close()


def normalize_null_eigens(cfile):

    fit = fc.Fit()

    print("Reading the configuration file & data.")
    fit.read_config(cfile)
    cfg = fit.cfg

    print("Setting current directory")
    subdir = cfg.folder
    outdir = os.path.join(cfg.outdir, subdir)

    norm_eigen_path = os.path.join(outdir,"stored-norm-null-eigens")
    
    if os.path.exists(norm_eigen_path):
        use_stored_eigeny = input("Would you like to use the stored normalized null eigen results? (y/n) ")

        if use_stored_eigeny.lower().strip() == "y":
            try:
                return np.loadtxt(f"{norm_eigen_path}/norm_null_eigeny.txt")
            except:
                print("File does not exist, creating a new version.")
    
    else:
        os.mkdir(norm_eigen_path)


    print("\nCalling create_eigens")
    print("----------------------------------------------------------------------------")
    eigeny, evalues, evectors, ecurves, lcs, star, cfg, fit = \
        create_eigens.create_eigens(cfile)
    

    print("Creating a new star with rv=True")
    lmax = cfg.sim.lmax
    udeg = cfg.star.udeg
    nsamples = cfg.sim.nsamples
    rv_star = utils.initstar(fit, lmax, udeg=udeg, include_rv=True)

    theta = np.linspace(0, 360, 60)
    
    ecurve_A = ecurves.T

    ecurve_R = np.empty((nsamples, lmax))

    for k in range(nsamples):
        
        ecurve_R[k] = [
            np.linalg.matrix_rank(ecurve_A[:, : (l + 1) ** 2]) for l in range(1, lmax + 1)
        ]

    ecurve_R = np.median(ecurve_R, axis=0) 

    null_eigeny = eigeny[int(ecurve_R[-1]):]

    print("Normalizing Null Maps to be Centered at 0 Flux")
    for i in range(null_eigeny.shape[0]):

        rv_star.map[0,0] = 0
        rv_star.map[1:,:] = 0
        yi = 1
        for l in range(1, lmax + 1):
            for mn in range(-l, l + 1):
                rv_star.map[l, mn] = null_eigeny[i, yi]
                yi += 1

        correction_f = rv_star.map.flux(theta=theta).eval()[0]
        null_eigeny[i,0] = -correction_f


    print("Saving New Normalization")
    np.savetxt(f"{norm_eigen_path}/norm_null_eigeny.txt", null_eigeny)

    return null_eigeny



def set_up_stars(cfile, eigeny, uni_comp = 1, theta_neg_90 = np.linspace(-90,90,720)):

    fit = fc.Fit()

    print("Reading the configuration file & data.")
    fit.read_config(cfile)
    cfg = fit.cfg

    print("Creating needed stars for Scaling")
    lmax = cfg.sim.lmax
    udeg = cfg.star.udeg


    #load in data/set up stars
    realistic_star = utils.initstar(fit, lmax, udeg=udeg, include_rv=True)
    base_star = utils.initstar(fit, lmax, include_rv=True)

    inv_realistic_star = utils.initstar(fit, lmax, udeg=udeg, include_rv=True)
    inv_base_star = utils.initstar(fit, lmax, include_rv=True)

    uni_star_limb = utils.initstar(fit, lmax, udeg=udeg, include_rv=True)
    uni_star = utils.initstar(fit, lmax, include_rv=True)

    #reset the star, create version without limb darkening, and create versions of unfirom stars
    realistic_star.map[0,0] = eigeny[0] + uni_comp
    realistic_star.map[1:,:] = eigeny[1:]

    inv_realistic_star.map[0,0] = -eigeny[0] + uni_comp
    inv_realistic_star.map[1:,:] = -eigeny[1:]


    #central intenisty is the same regardless of inc, so setting to 90 for ease
    uni_star.map.inc = 90

    uni_star_limb.map.inc = 90

    uni_star.map[0,0] = uni_comp
    uni_star.map[1:,:] = 0

    uni_star_limb.map[0,0] = uni_comp
    uni_star_limb.map[1:,:] = 0

    uni_I = uni_star.map.intensity(lat=0, lon=theta_neg_90,rv=False).eval()
    uni_I_limb = uni_star_limb.map.intensity(lat=0, lon=theta_neg_90,rv=False).eval()

    ratio_no_limb = np.max(uni_I) / np.max(uni_I_limb)

    return realistic_star, inv_realistic_star, uni_star_limb, uni_star, ratio_no_limb



def set_up_real_map(realistic_star, inv_realistic_star, uni_star_limb, ratio_no_limb, 
                    uni_comp = 1, theta_neg_90 = np.linspace(-90,90,720),
                    lower_fraction = .42, upper_fraction = 1.35):

    uni_star_limb.map[0,0] = uni_comp * ratio_no_limb
    uni_star_limb.map[1:,:] = 0
    uni_value = np.max(uni_star_limb.map.intensity(lat=0, lon=theta_neg_90,rv=False).eval())


    amount_below_uni = uni_value - realistic_star.map.minimize()[-1].eval()
    amount_above_uni = uni_value - inv_realistic_star.map.minimize()[-1].eval()

    ratio_below = (uni_value - amount_below_uni) / uni_value
    ratio_above = (amount_above_uni + uni_value) / uni_value

    print("Initial Values:")
    print("----------------------------------------------------------------------------")
    print("\tAmount uni/below/above:",round(uni_value,2),round(amount_below_uni,2),round(amount_above_uni,2))
    print("\tRatio below/above:",round(ratio_below,2),round(ratio_above,2))
    print("\tValue uni/below/above:",round(uni_value,2),round(ratio_below*uni_value,2),round(ratio_above*uni_value,2))
    print("----------------------------------------------------------------------------")
    

    while ratio_below <= lower_fraction or ratio_above >= upper_fraction:

        """
        Note: starry does not apply limb darkening to render or minimize.  
        So while these maps need to be scaled for .show() animations and intensity curves, 
        they do not need to be scaled when deadling with static plots/minimize/anything that
        can't have limb darkening applied
        """

        uni_comp += .1

        uni_star_limb.map[0,0] = uni_comp * ratio_no_limb
        uni_star_limb.map[1:,:] = 0
        uni_value = np.max(uni_star_limb.map.intensity(lat=0, lon=theta_neg_90,rv=False).eval())

        realistic_star.map[0,0] += .1

        inv_realistic_star.map[0,0] += .1

        amount_below_uni = uni_value - realistic_star.map.minimize()[-1].eval()
        amount_above_uni = uni_value - inv_realistic_star.map.minimize()[-1].eval()

        ratio_below = (uni_value - amount_below_uni) / uni_value
        ratio_above = (amount_above_uni + uni_value) / uni_value

        # print("Amount uni/below/above:",round(uni_value,2),round(amount_below_uni,2),round(amount_above_uni,2))
        # print("Ratio below/above:",round(ratio_below,2),round(ratio_above,2))
        # print("Value uni/below/above:",round(uni_value,2),round(ratio_below*uni_value,2),round(ratio_above*uni_value,2))


    print("Finished Normalizing with Values:")
    print("----------------------------------------------------------------------------")
    print("\tAmount uni/below/above:",round(uni_value,2),round(amount_below_uni,2),round(amount_above_uni,2))
    print("\tRatio below/above:",round(ratio_below,2),round(ratio_above,2))
    print("\tValue uni/below/above:",round(uni_value,2),round(ratio_below*uni_value,2),round(ratio_above*uni_value,2))
    print("----------------------------------------------------------------------------")

    """
    As described in the note above, we want to apply the amp after because limb only impacts intensity and animated plots
    Therefore if we apply it before we will have the same issue where the limb-calcs will be inflated unrealistically
    This means we also need to turn the amp "off" for any static plots/calc without limb involved.
    """

    print("Applying the ratio_no_limb to Star's Amp to fix Starry limb darkening.")
    realistic_star.map.amp = ratio_no_limb


    return realistic_star, uni_comp, ratio_below, ratio_above, uni_value

   


def generate_result_for_star(realistic_star, uni_comp, ratio_below, ratio_above, uni_value, results_path, folder_name):

    corrected_amp = realistic_star.map.amp.eval()

    if not os.path.isdir(f"{results_path}/{folder_name}"):
        os.mkdir(f"{results_path}/{folder_name}")

    eigeny = realistic_star.map.y.eval()

    data_array = np.array([uni_comp, uni_value, ratio_below, ratio_above, corrected_amp])

    np.savetxt(f"{results_path}/{folder_name}/null_map_eigeny.txt", eigeny)
    np.savetxt(f"{results_path}/{folder_name}/null_map_data.txt", data_array)

    theta = np.linspace(0, 360, 60)

    # image = realistic_star.map.render(projection="rect",rv=True).eval()

    realistic_star.map.show(theta=theta,rv=False, file=f"{results_path}/{folder_name}/flux_map.mp4", dpi = 300)
    realistic_star.map.show(theta=theta,rv=True, file=f"{results_path}/{folder_name}/rv_map.mp4", dpi = 300)
    
    create_rv.rv_flux_map_ani(realistic_star,theta,f"{results_path}/{folder_name}/both_maps.mp4")


    # NOTE TO SELF: May want to make this a normal amp map since no limb darkening (so set it to be 1, then back to ratio)
    # Also fix dpi issue

    realistic_star.map.amp = 1
    plt.figure(figsize=(12, 5))
    image = realistic_star.map.render(projection="rect",rv=False).eval()
    plt.imshow(image, origin="lower", cmap="plasma", extent=(-180, 180, -90, 90))
    plt.xlabel("longitude [deg]")
    plt.ylabel("latitude [deg]")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f"{results_path}/{folder_name}/smooth_flux_rect.png", dpi = 300)
    plt.close()


    realistic_star.map.show(projection="rect",rv=False,colorbar=True,file=f"{results_path}/{folder_name}/flux_rect.png", dpi=300)

    realistic_star.map.amp = corrected_amp

    create_rv.flux_rv_line(realistic_star,theta,
                    flux_name=f"{results_path}/{folder_name}/flux_curve.png",rv_name=f"{results_path}/{folder_name}/rv_curve.png")

    create_rv.multi_phase_plot(realistic_star,fname=f"{results_path}/{folder_name}/map_slideshow.png")


    create_rv.map_curve_ani(realistic_star,theta,fname=f"{results_path}/{folder_name}/animated_plots.mp4")

    create_rv.map_curve_ani(realistic_star,theta,fname=f"{results_path}/{folder_name}/animated_plots.gif")


    plot_intensity(realistic_star,theta,fname=f"{results_path}/{folder_name}/intensity_animation.mp4")


    theta_i = np.linspace(-180,180,360)
    mu = np.cos((theta_i) * np.pi/180)

    realistic_star.map.amp = ratio_no_limb
    star_intensity = realistic_star.map.intensity(lat=0, lon=theta_i,rv=False).eval()

    realistic_star.map.amp = 1
    no_limb_intenisty = realistic_star.map.intensity(lat=0, lon=theta_i,rv=False,limbdarken=False).eval()

    udeg = realistic_star.map.u.eval()
    limb_law = 1 - udeg[0]*(1-mu) - udeg[1] * (1-mu)**2


    plt.figure(figsize=(12, 5))
    plt.plot(theta_i,star_intensity,label="$I_{limb}$",alpha=1)
    plt.plot(theta_i, no_limb_intenisty, ls=":",alpha=.5,label="$I_{no limb}$",c="k")
    plt.xlabel("longitude [deg]", fontsize=24)
    plt.ylabel("Intensity", fontsize=24)
    plt.tight_layout()
    plt.legend()
    plt.savefig(f"{results_path}/{folder_name}/intensity_curve.png", dpi = 300)
    plt.close()
    
    
    print(f"{folder_name} plots are complete!")
    print("----------------------------")

    






if __name__ == "__main__":
    """
    """
    # print(sys.argv) #Uncomment if you want to see command line arguments 
    if len(sys.argv) < 2:
        print("ERROR: Call structure is python real_null_maps.py <configuration file>.")
        sys.exit()
    else:
        cfile = sys.argv[1]

    null_eigens = normalize_null_eigens(cfile)

    fit = fc.Fit()

    fit.read_config(cfile)
    cfg = fit.cfg

    print("Setting current directory")
    subdir = cfg.folder
    outdir = os.path.join(cfg.outdir, subdir)
    results_path = os.path.join(outdir,"realistic_outputs")

    if not os.path.isdir(results_path):
        os.mkdir(results_path)

    for i in range(null_eigens.shape[0]):

        realistic_star, inv_realistic_star, uni_star_limb, uni_star, ratio_no_limb = set_up_stars(cfile, null_eigens[i])

        realistic_star, uni_comp, ratio_below, ratio_above, uni_value = \
            set_up_real_map(realistic_star, inv_realistic_star, uni_star_limb, ratio_no_limb)
    

        generate_result_for_star(realistic_star, uni_comp, ratio_below, ratio_above, uni_value, results_path, folder_name = f"null_map_{i}")

    for i in range(null_eigens.shape[0]):

        realistic_star, inv_realistic_star, uni_star_limb, uni_star, ratio_no_limb = set_up_stars(cfile, -null_eigens[i])

        realistic_star, uni_comp, ratio_below, ratio_above, uni_value = \
            set_up_real_map(realistic_star, inv_realistic_star, uni_star_limb, ratio_no_limb)
    

        generate_result_for_star(realistic_star, uni_comp, ratio_below, ratio_above, uni_value, results_path, folder_name = f"null_map_-{i}")

    print("done")
    sys.exit()




#EXTRA CODE


 # theta = np.linspace(0, 360, 60)

    # eigeny, evalues, evectors, ecurves, lcs, star, cfg, fit = \
    #     create_eigens.create_eigens(cfile)
    

    # print("Creating a new star with rv=True")
    # lmax = cfg.sim.lmax
    # udeg = cfg.star.udeg
    # nsamples = cfg.sim.nsamples
    # rv_star = utils.initstar(fit, lmax, udeg=udeg, include_rv=True)

    # for eig in null_eigens:

    #     rv_star.map[:,:] = eig

    #     rv_star.map[0,0] += 1

    #     print(eig)
    #     print(rv_star.map.y.eval())

    #     create_rv.flux_rv_line(rv_star,theta,flux_name=None,rv_name=None,flux_only=True,rv_only=False)

    #     pass
