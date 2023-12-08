import numpy as np
from scipy.interpolate import LinearNDInterpolator
from threeML import PhAbs, Powerlaw

# LogN-LogS parameters from Moretti et al. 2003
cxb_m13_unres = 1.9411e-07  # unresolved CXB fraction at a limiting flux of 1e-15 in the soft band from Moretti+03
alpha1 = 1.82
alpha2 = 0.60
s0 = 1.48e-14
Ns = 6150.
B = s0 ** (alpha1 - alpha2)
pivot = 2e-15
N_gt15 = Ns * pivot ** alpha1 / (1e-15 ** alpha1 + B * 1e-15 ** alpha2)  # per square degree

gamma_mean = 1.9  # Normal distribution of photon indices
gamma_sig = 0.2

nh_min = 20.5  # Simulating a flat distribution of NH from 20.5 to 23; larger NH is usually not observable
nh_max = 23.0
flux_norm_conv = 2.2211e-09  # flux corresponding to a normalization of 1.0


def dNdS(S):
    '''
    Differential logN-logS function from Moretti+03

    :param S: Source flux in [0.5-2] keV band
    :return: dN/dS per square degree
    '''
    num = alpha2 * B * S ** alpha2 + alpha1 * S ** alpha1

    denom = S * (B * S ** alpha2 + S ** alpha1) ** 2

    return - Ns * pivot ** alpha1 * num / denom


def gen_sources(xmmsim, outfile):
    '''
    Create a distribution of fluxes, photon indices and NH, place them randomly within the field and save them to output file

    :param xmmsim:
    :param outfile:
    :return:
    '''

    field_area = (xmmsim.box_size / 60.) ** 2

    n_src = np.random.poisson(N_gt15 * field_area)

    npix = xmmsim.box.shape[0]

    S_grid_15 = np.logspace(-15, -12, 1000)  # cutting at -12 as greater values are highly improbable in a 0.25 square deg area

    prob_dist = dNdS(S_grid_15) / np.sum(dNdS(S_grid_15))

    sel_sources = np.random.choice(S_grid_15, size=n_src, p=prob_dist)  # source fluxes

    X_src = np.random.rand(n_src) * npix
    Y_src = np.random.rand(n_src) * npix

    gamma_src = gamma_mean + gamma_sig * np.random.randn(n_src)

    NH_src = nh_min + (nh_max - nh_min) * np.random.rand(n_src)

    srcnum = np.arange(1, n_src + 1)

    np.savetxt(outfile, np.transpose([srcnum, X_src, Y_src, sel_sources, gamma_src, NH_src]),
               header='srcnum X Y Fx Gamma NH')


def pts_box(xmmsim, source_file):
    '''
    Generate a box of spectra for point sources on the same grid as the initial simulation

    :param xmmsim:
    :param sources:
    :return:
    '''

    # Get template file
    template_file = get_data_file_path('pts/templates_pts.dat')

    template = np.loadtxt(template_file)

    fint = LinearNDInterpolator(template[:, :2], template[:, 2])

    # Read source file
