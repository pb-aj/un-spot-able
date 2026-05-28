"""
[Description needed here]
https://github.com/pb-aj/un-spot-able
"""

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
from lib import utils
from lib import fitclass    as fc
from lib.spotable import spotable as se

# py imports
import create_eigens
import create_rv
import create_eigens

# Set up starry configuration
starry.config.quiet = True
starry.config.lazy = True
sys.setrecursionlimit(10000) # starry seems to have a lot of recursion


# Whether or not to show print statements in code
dpm = True #change to False to make quiet

def normalize_null_eigens(cfile, prompt_user=True):

    fit = fc.Fit()

    se("\tReading the configuration file & data",dp = dpm)

    try:
        fit.read_config(cfile)
    except:
        se("----------------------------------------------------------------------------",dp = dpm)
        sys.exit('\033[31mERROR:\033[0m' + ' Check Name of Configuration File.',dp = dpm)
    
    cfg = fit.cfg

    subdir = cfg.folder
    outdir = os.path.join(cfg.outdir, subdir)
    norm_eigen_path = os.path.join(outdir,"stored-norm-null-eigens")

    se(f"\n\tSetting directory to:\n\t\033[34m{outdir}\033[0m\n",dp = dpm)

    if prompt_user:
        if os.path.exists(norm_eigen_path):
            use_stored_eigeny = input("\tWould you like to use the stored normalized null eigen results? (y/n) ")
        else:
            se("\tNo stored normalized null eigen results found, creating new ones...",dp = dpm)
            os.mkdir(norm_eigen_path)
            use_stored_eigeny = "n"
    else:
        if os.path.exists(norm_eigen_path):
            use_stored_eigeny = "y"
        else:
            se("\tNo stored normalized null eigen results found, creating new ones...",dp = dpm)
            os.mkdir(norm_eigen_path)
            use_stored_eigeny = "n"

    if use_stored_eigeny.lower().strip() == "y":
        if prompt_user:
            se("",dp=dpm)
            
        se(f"\tReading previously stored normalized null eigen results from:\n\t\033[34m{norm_eigen_path}\033[0m",dp = dpm)
        try:
            norm_eigeny = np.loadtxt(f"{norm_eigen_path}/norm_null_eigeny.txt")
            se("----------------------------------------------------------------------------", dp = dpm)

            return norm_eigeny
        except:
            se(f"\n\t\033[31m\033[1mStored normalized null eigen results are invalid, calculating new ones...\033[0m",dp = dpm)
            use_stored_eigeny = "n"
    elif prompt_user:
        se("\n\tGenerating new normalized null eigen results...",dp = dpm)
    se("----------------------------------------------------------------------------", dp = dpm)

    se("\n\033[32mCalling create_eigens to create initial maps:\033[0m", dp = dpm)
    se("----------------------------------------------------------------------------", dp = dpm)
    eigeny, evalues, evectors, ecurves, lcs, star, fit = \
        create_eigens.create_eigens(cfile,prompt_user=prompt_user)
    

    se("\n\033[32mNormalizing Null Maps:\033[0m", dp = dpm)
    se("----------------------------------------------------------------------------", dp = dpm)   

    se("\tCreating a new star with rv=True", dp = dpm)
    lmax = cfg.sim.lmax
    udeg = cfg.star.udeg

    rv_star = utils.initstar(fit, lmax, udeg=udeg, include_rv=True)

    se("\n\tFinding rank of the rv star design matrix:", dp = dpm)

    A = star.map.design_matrix(theta=np.linspace(0,360,100)).eval()
    R = np.empty((1, lmax))

    R = [
        np.linalg.matrix_rank(A[:, : (l + 1) ** 2]) for l in range(1, lmax + 1)
    ]

    #Display rank for each spherical harmonic degree
    se("\t------------------------------------------------",dp = dpm)
    se(f"\t\033[1mThe rank (# non-null maps) for lmax = {lmax} (ncurves = {int(A.shape[1])}) is {int(R[-1])}\033[0m",dp = dpm) 
    se("\t------------------------------------------------\n",dp = dpm)

    se("\tUsing rank to pull out only null space maps", dp = dpm)
    null_eigeny = eigeny[int(R[-1]):]

    se("\n\tCentering null maps at 0 Flux", dp = dpm)
    for i in range(null_eigeny.shape[0]):

        rv_star.map[:,:] = null_eigeny[i]
        rv_star.map[0,0] = 0

        correction_f = rv_star.map.flux(theta=np.linspace(0,360,60)).eval()
        null_eigeny[i,0] = -np.median(correction_f)


    se(f"\n\tStoring normalized null maps to:\n\t\033[34m{norm_eigen_path}\033[0m", dp = dpm)
    np.savetxt(f"{norm_eigen_path}/norm_null_eigeny.txt", null_eigeny)

    se("----------------------------------------------------------------------------", dp = dpm)   

    return null_eigeny


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


def set_up_stars(cfile, eigeny, uni_comp = 1, theta_neg_90 = np.linspace(-90,90,720)):

    fit = fc.Fit()

    se("Reading the configuration file & data.", dp = dpm)
    fit.read_config(cfile)
    cfg = fit.cfg

    se("Creating needed stars for Scaling", dp = dpm)
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

    se("Initial Values:", dp = dpm)
    se("----------------------------------------------------------------------------", dp = dpm)
    se("\tAmount uni/below/above:",round(uni_value,2),round(amount_below_uni,2),round(amount_above_uni,2), dp = dpm)
    se("\tRatio below/above:",round(ratio_below,2),round(ratio_above,2), dp = dpm)
    se("\tValue uni/below/above:",round(uni_value,2),round(ratio_below*uni_value,2),round(ratio_above*uni_value,2), dp = dpm)
    se("----------------------------------------------------------------------------")
    

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

        # se("Amount uni/below/above:",round(uni_value,2),round(amount_below_uni,2),round(amount_above_uni,2))
        # se("Ratio below/above:",round(ratio_below,2),round(ratio_above,2))
        # se("Value uni/below/above:",round(uni_value,2),round(ratio_below*uni_value,2),round(ratio_above*uni_value,2))


    se("Finished Normalizing with Values:", dp = dpm)
    se("----------------------------------------------------------------------------", dp = dpm)
    se("\tAmount uni/below/above:",round(uni_value,2),round(amount_below_uni,2),round(amount_above_uni,2), dp = dpm)
    se("\tRatio below/above:",round(ratio_below,2),round(ratio_above,2), dp = dpm)
    se("\tValue uni/below/above:",round(uni_value,2),round(ratio_below*uni_value,2),round(ratio_above*uni_value,2), dp = dpm)
    se("----------------------------------------------------------------------------", dp = dpm)

    """
    As described in the note above, we want to apply the amp after because limb only impacts intensity and animated plots
    Therefore if we apply it before we will have the same issue where the limb-calcs will be inflated unrealistically
    This means we also need to turn the amp "off" for any static plots/calc without limb involved.
    """

    se("Applying the ratio_no_limb to Star's Amp to fix Starry limb darkening.")
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
    
    
    se(f"{folder_name} plots are complete!")
    se("----------------------------")

    



if __name__ == "__main__":
    # Uncomment if you want to see command line arguments 
    # se(sys.argv,dp = dpm) 

    # Check command line input is correct
    if len(sys.argv) < 2:
        se("\n----------------------------------------------------------------------------",dp = dpm)
        se('\033[31mERROR:\033[0m' + ' Call structure is "\033[34mpython create_real_null.py <configuration file>\033[0m"',dp = dpm)
        se("----------------------------------------------------------------------------",dp = dpm)
        sys.exit()
    else:
        cfile = sys.argv[1]

    se("\n\033[32mNormailizing Null Maps:\033[0m", dp = dpm)
    se("----------------------------------------------------------------------------", dp = dpm)
    null_eigens = normalize_null_eigens(cfile, prompt_user=True)

    se("\n\033[32mSetting up Calibration Stars:\033[0m", dp = dpm)
    se("----------------------------------------------------------------------------", dp = dpm)
    



    sys.exit("\033[32mdone\033[0m")




# if __name__ == "__main__":
#     """
#     """
#     # se(sys.argv, dp = dpm) #Uncomment if you want to see command line arguments 
#     if len(sys.argv) < 2:
#         se("ERROR: Call structure is python real_null_maps.py <configuration file>.", dp = dpm)
#         sys.exit()
#     else:
#         cfile = sys.argv[1]

#     null_eigens = normalize_null_eigens(cfile)

#     sys.exit()

#     fit = fc.Fit()

#     fit.read_config(cfile)
#     cfg = fit.cfg

#     se("Setting current directory", dp = dpm)
#     subdir = cfg.folder
#     outdir = os.path.join(cfg.outdir, subdir)
#     results_path = os.path.join(outdir,"realistic_outputs")

#     if not os.path.isdir(results_path):
#         os.mkdir(results_path)

#     for i in range(null_eigens.shape[0]):

#         realistic_star, inv_realistic_star, uni_star_limb, uni_star, ratio_no_limb = set_up_stars(cfile, null_eigens[i])

#         realistic_star, uni_comp, ratio_below, ratio_above, uni_value = \
#             set_up_real_map(realistic_star, inv_realistic_star, uni_star_limb, ratio_no_limb)
    

#         generate_result_for_star(realistic_star, uni_comp, ratio_below, ratio_above, uni_value, results_path, folder_name = f"null_map_{i}")

#     for i in range(null_eigens.shape[0]):

#         realistic_star, inv_realistic_star, uni_star_limb, uni_star, ratio_no_limb = set_up_stars(cfile, -null_eigens[i])

#         realistic_star, uni_comp, ratio_below, ratio_above, uni_value = \
#             set_up_real_map(realistic_star, inv_realistic_star, uni_star_limb, ratio_no_limb)
    

#         generate_result_for_star(realistic_star, uni_comp, ratio_below, ratio_above, uni_value, results_path, folder_name = f"null_map_-{i}")

#     se("done", dp = dpm)
#     sys.exit()




#EXTRA CODE


 # theta = np.linspace(0, 360, 60)

    # eigeny, evalues, evectors, ecurves, lcs, star, cfg, fit = \
    #     create_eigens.create_eigens(cfile)
    

    # se("Creating a new star with rv=True")
    # lmax = cfg.sim.lmax
    # udeg = cfg.star.udeg
    # nsamples = cfg.sim.nsamples
    # rv_star = utils.initstar(fit, lmax, udeg=udeg, include_rv=True)

    # for eig in null_eigens:

    #     rv_star.map[:,:] = eig

    #     rv_star.map[0,0] += 1

    #     se(eig)
    #     se(rv_star.map.y.eval())

    #     create_rv.flux_rv_line(rv_star,theta,flux_name=None,rv_name=None,flux_only=True,rv_only=False)

    #     pass
