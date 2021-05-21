import numpy as np
from astropy.io import fits
from .utils import get_data_file_path, calc_arf
from scipy.interpolate import interp2d
from scipy.signal import convolve

def psf_convole(image, pixsize, xmmsim):
    """
    Convolve an input image with the telescope's on-axis PSF

    :param image:
    :param xmmsim:
    :return:
    """

    fpsf = fits.open(xmmsim.ccfpath + xmmsim.psf_file)

    king_params = fpsf['KING_PARAMS'].data

    onaxis_params = king_params['PARAMS'][0] # Sticking to on-axis, outside doesn't matter so much

    r0 = onaxis_params[0] / 60.  # core radius in arcmin

    alpha = onaxis_params[1] # outer slope

    fpsf.close()

    npix_out = image.shape[0]

    y, x = np.indices(image.shape)

    cx, cy = npix_out / 2., npix_out / 2.

    rads = np.hypot(x - cx, y - cy) * pixsize  # arcmin

    kernel = np.power(1. + (rads / r0) ** 2, - alpha) # Convolution kernel

    norm = np.sum(kernel)

    kernel = kernel / norm

    # FFT-convolve image with kernel
    blurred = convolve(image, kernel, mode='same')

    return blurred


def gen_image_box(xmmsim, tsim, elow=0.5, ehigh=2.0, nbin=10):
    """
    Generate an image of counts/pixel expectation values in an input energy band

    :param xmmsim: XMMSimulator
    :param tsim: Simulation exposure time in sec
    :param elow: Lower energy boundary of the image (defaults to 0.5)
    :param ehigh: Upper energy boundary of the image (defaults to 2.0)
    :param nbin: Number of energy bins to subdivide the band for vignetting curve calculation
    :return:
        - Image of counts/pixel
        - Pixel size in arcmin
    """

    # Get mask file
    mask_file = get_data_file_path('imgs/%s_mask.fits.gz' % (xmmsim.instrument))

    inmask = fits.open(mask_file)

    mask = inmask[1].data

    pixsize = inmask[1].header['CDELT2'] * 60.  # arcmin

    npix_out = mask.shape[0]

    ene = xmmsim.box_ene

    xori = np.arange(0, xmmsim.box.shape[1], 1)
    yori = np.arange(0, xmmsim.box.shape[0], 1)

    newima = np.zeros(mask.shape)

    cx, cy = npix_out / 2., npix_out / 2.

    y, x = np.indices(mask.shape)

    thetas = np.hypot(x - cx, y - cy) * pixsize  # arcmin

    ene_bins = np.linspace(elow, ehigh, nbin+1)

    ene_bin_width = ene_bins[1] - ene_bins[0]

    arf_onaxis = calc_arf(0., ene_bins[:nbin], ene_bins[1:], xmmsim)

    # Reading vignetting curve
    dtheta = xmmsim.dtheta

    rads = np.arange(0., 16., dtheta)

    ene_vig = xmmsim.vignetting['ENERGY'] / 1e3
    vig_fact = xmmsim.vignetting['VIGNETTING_FACTOR']

    for i in range(nbin):
        # Select box data in the chosen energy band
        eband = np.where(np.logical_and(ene >= ene_bins[i], ene < ene_bins[i+1]))

        ima = np.sum(xmmsim.box[:, :, eband], axis=3)

        # Recast box shape into output image shape
        finterp = interp2d(xori, yori, ima[:, :, 0], kind='linear')

        xnew = np.arange(0, npix_out, 1) * ima.shape[1] / npix_out

        ynew = np.arange(0, npix_out, 1) * ima.shape[0] / npix_out

        ima_newpix = finterp(xnew, ynew) * ima.shape[0] * ima.shape[1] / (npix_out ** 2) # phot/cm2/s/keV

        # Compute vignetting curve
        ene_bin = (ene_bins[i] + ene_bins[i + 1]) / 2.

        nearest = np.argsort(np.abs(ene_vig - ene_bin))[0]

        vig_curve = vig_fact[nearest]

        texp = np.interp(thetas, rads, vig_curve) * tsim

        newima = newima + arf_onaxis[i] * ima_newpix * texp * xmmsim.box_ene_width

    blurred = psf_convole(newima, pixsize, xmmsim)

    outima = blurred * mask

    return outima

