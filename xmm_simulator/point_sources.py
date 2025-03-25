import numpy as np
from scipy.interpolate import LinearNDInterpolator
from. utils import get_data_file_path, set_wcs
from threeML import PhAbs, Powerlaw

# LogN-LogS parameters from Moretti et al. 2003
cxb_m13_unres = 1.9411e-07  # unresolved CXB fraction at a limiting flux of 1e-15 in the soft band from Moretti+03
alpha1 = 1.57 # hard-band logN-logS
alpha2 = 0.44
s0 = 4.5e-15
Ns = 5300

B = s0 ** (alpha1 - alpha2)
pivot = 2e-15
N_gt15 = Ns * pivot ** alpha1 / (1e-15 ** alpha1 + B * 1e-15 ** alpha2)  # per square degree

gamma_mean = 1.9  # Normal distribution of photon indices
gamma_sig = 0.2

nh_min = 20.5  # Simulating a flat distribution of NH from 20.5 to 23; larger NH is usually not observable
nh_max = 23.0

def dNdS(S):
    '''
    Differential logN-logS function from Moretti+03

    :param S: Source flux in [0.5-2] keV band
    :return: dN/dS per square degree
    '''
    num = alpha2 * B * S ** alpha2 + alpha1 * S ** alpha1

    denom = S * (B * S ** alpha2 + S ** alpha1) ** 2

    return - Ns * pivot ** alpha1 * num / denom


def gen_sources(xmmsim, outfile=None, outreg=None):
    '''
    Create a distribution of fluxes, photon indices and NH, place them randomly within the field and save them to output file

    :param xmmsim:
    :param outfile:
    :param outreg:
    :return:
    '''

    if outfile is None:
        print('No output file provided, aborting')
        return

    field_area = (xmmsim.box_size / 60.) ** 2

    n_src = np.random.poisson(N_gt15 * field_area)

    #npix = xmmsim.box.shape[0]
    npix = xmmsim.boxshape0

    S_grid_15 = np.logspace(-15, -12, 1000)  # cutting at -12 as greater values are highly improbable in a 0.25 square deg area

    dS = (np.log10(S_grid_15[1]) - np.log10(S_grid_15[0])) * S_grid_15 * np.log(10.)

    unnorm = dNdS(S_grid_15) * dS

    prob_dist = unnorm / np.sum(unnorm)

    sel_sources = np.random.choice(S_grid_15, size=n_src, p=prob_dist)  # source fluxes

    X_src = np.random.rand(n_src) * npix
    Y_src = np.random.rand(n_src) * npix

    gamma_src = gamma_mean + gamma_sig * np.random.randn(n_src)

    NH_src = nh_min + (nh_max - nh_min) * np.random.rand(n_src)

    redshifts = np.random.rand(n_src) * 1.5 # assuming flat redshift distribution out to 1.5

    srcnum = np.arange(1, n_src + 1)

    wcs = set_wcs(xmmsim, type='box')

    pixcrd = np.array([X_src+1,Y_src+1]).T

    wc = wcs.wcs_pix2world(pixcrd, 1)

    rasrc = wc[:,0]

    decsrc = wc[:,1]

    np.savetxt(outfile, np.transpose([srcnum, X_src, Y_src, sel_sources, gamma_src, NH_src, redshifts, rasrc, decsrc]),
               header='srcnum X Y Fx Gamma NH z RA Dec')

    if outreg is not None:
        freg = open(outreg, 'w')
        freg.write('# Region file format: DS9 version 4.1\n')
        freg.write('global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n')
        freg.write('fk5\n')
        npts = len(X_src)
        for i in range(npts):
            freg.write('circle(%g, %g, 30")\n' % (rasrc[i], decsrc[i]))

        freg.close()


def pts_box(xmmsim, source_file):
    '''
    Generate a box of spectra for point sources on the same grid as the initial simulation

    :param xmmsim:
    :param sources:
    :return:
    '''

    # Get template file
    template_file = get_data_file_path('pts/templates_pts_hard.dat')

    template = np.loadtxt(template_file)

    fint = LinearNDInterpolator(template[:, :2], template[:, 2], fill_value=np.mean(template[:,2]))

    # Read source file
    pts = np.loadtxt(source_file)

    X_pts = pts[:,1]
    Y_pts = pts[:,2]
    Fx_pts = pts[:,3]
    Gamma_pts = pts[:,4]
    NH_pts = pts[:,5]
    z_pts = pts[:,6]

    flux_norm_conv = fint(Gamma_pts, NH_pts)

    norm = Fx_pts / flux_norm_conv

    #box_pts = np.zeros(xmmsim.box.shape)
    box_pts = np.zeros((xmmsim.boxshape0, xmmsim.boxshape1, xmmsim.boxshape2))

    npts = len(X_pts)

    mod = PhAbs() * Powerlaw()

    for i in range(npts):

        mod.NH_1 = 10 ** (NH_pts[i] - 22.)
        mod.index_2 = -Gamma_pts[i]
        mod.K_2 = norm[i]
        mod.redshift_1 = z_pts[i]

        iX = int(X_pts[i] + 0.5)
        iY = int(Y_pts[i] + 0.5)

        #if iX==xmmsim.box.shape[1]: iX = xmmsim.box.shape[1] - 1
        #if iY==xmmsim.box.shape[0]: iY = xmmsim.box.shape[0] - 1
        if iX==xmmsim.boxshape1: iX = xmmsim.boxshape1 - 1
        if iY==xmmsim.boxshape0: iY = xmmsim.boxshape0 - 1

        box_pts[iY, iX, :] = mod(xmmsim.box_ene_mean) * xmmsim.tsim * xmmsim.all_arfs[iY, iX, :]

    xmmsim.box_pts = box_pts
    xmmsim.pts = True


