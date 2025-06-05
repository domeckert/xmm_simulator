import numpy as np
from astropy.io import fits
from .utils import get_data_file_path, calc_arf
from scipy.interpolate import interp2d
from scipy.signal import convolve
from datetime import datetime
import os
from threeML.utils.OGIP.response import OGIPResponse

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



def psf_convole_evt(Xevt, Yevt, pixsize, xmmsim):
    """
    Convolve an event file list with the telescope's on-axis PSF
    :param Xevt:
    :param Yevt:
    :param xmmsim:
    :return:
    """

    fpsf = fits.open(xmmsim.ccfpath + xmmsim.psf_file)

    king_params = fpsf['KING_PARAMS'].data

    onaxis_params = king_params['PARAMS'][0] # Sticking to on-axis, outside doesn't matter so much

    r0 = onaxis_params[0] / 60.  # core radius in arcmin

    alpha = onaxis_params[1] # outer slope

    fpsf.close()

    num_photons = len(Xevt)  # Number of photons

    u = np.random.uniform(0, 1, num_photons)  # Uniform random numbers
    #Analytical method
    #Draw offsets from the King's CDF
    #r_samples = r0 * np.sqrt((1 / u) ** (1 / (alpha - 1)) - 1)

    #Numerical method
    offsets = np.geomspace(1e-4, 50, 1000)
    king_profile = (1 + (offsets / r0) ** 2) ** (-alpha)
    king_cdf = np.cumsum(king_profile)/np.sum(king_profile)
    r_samples = np.interp(u, king_cdf, offsets)

    theta_samples = np.random.uniform(0, 2 * np.pi, size=num_photons)  # Random angles

    # Convert to Cartesian offsets
    delta_x = r_samples * np.cos(theta_samples) / pixsize  # Convert arcmin to pixels
    delta_y = r_samples * np.sin(theta_samples) / pixsize  # Convert arcmin to pixels

    # Apply the PSF effect: shift the photon positions
    Xpix_blurred = np.round(Xevt + delta_x).astype(int)
    Ypix_blurred = np.round(Yevt + delta_y).astype(int)

    return Xpix_blurred, Ypix_blurred

def exposure_map(xmmsim, tsim, elow=0.5, ehigh=2.0, nbin=10):
    """
    Generate an effective exposure map including vignetting curve in an input energy band

    :param xmmsim: XMMSimulator
    :param tsim: Simulation exposure time in sec
    :param elow: Lower energy boundary of the image (defaults to 0.5)
    :param ehigh: Upper energy boundary of the image (defaults to 2.0)
    :param nbin: Number of energy bins to subdivide the band for vignetting curve calculation
    :return: effective exposure map
    """
    # Get mask file
    mask_file = get_data_file_path('imgs/%s_mask.fits.gz' % (xmmsim.instrument))

    inmask = fits.open(mask_file)

    mask = inmask[1].data

    pixsize = inmask[1].header['CDELT2'] * 60.  # arcmin

    inmask.close()

    # Reading vignetting curve
    dtheta = xmmsim.dtheta

    rads = np.arange(0., 16., dtheta)

    ene_vig = xmmsim.vignetting['ENERGY'] / 1e3
    vig_fact = xmmsim.vignetting['VIGNETTING_FACTOR']

    ene_bins = np.linspace(elow, ehigh, nbin+1)

    ene_bin_width = ene_bins[1] - ene_bins[0]

    expmap, bkg_map = np.zeros(mask.shape), np.zeros(mask.shape)

    cx, cy = mask.shape[0] / 2., mask.shape[1] / 2.

    y, x = np.indices(mask.shape)

    thetas = np.hypot(x - cx, y - cy) * pixsize  # arcmin

    for i in range(nbin):

        ene_bin = (ene_bins[i] + ene_bins[i+1]) / 2.

        nearest = np.argsort(np.abs(ene_vig - ene_bin))[0]

        vig_curve = vig_fact[nearest]

        texp = np.interp(thetas, rads, vig_curve) * tsim

        expmap = expmap + texp

    expmap = expmap / nbin

    expmap = expmap * mask

    return expmap


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

    pixsize_ori = xmmsim.box_size / xmmsim.box.shape[1] # arcmin

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
        cx_ori, cy_ori =  xmmsim.box.shape[1]/2. , xmmsim.box.shape[0]/2.

        finterp = interp2d(yori - cy_ori, xori - cx_ori, ima[:, :, 0], kind='linear')

        xnew = (np.arange(0, npix_out, 1) - cx) * pixsize / pixsize_ori

        ynew = (np.arange(0, npix_out, 1) - cy) * pixsize / pixsize_ori

        ima_newpix = finterp(xnew, ynew) * (pixsize / pixsize_ori)**2 # phot/cm2/s/keV

        # Compute vignetting curve
        ene_bin = (ene_bins[i] + ene_bins[i + 1]) / 2.

        nearest = np.argsort(np.abs(ene_vig - ene_bin))[0]

        vig_curve = vig_fact[nearest]

        texp = np.interp(thetas, rads, vig_curve) * tsim

        newima = newima + arf_onaxis[i] * ima_newpix * texp * xmmsim.box_ene_width

    blurred = psf_convole(newima, pixsize, xmmsim)

    outima = blurred * mask

    return outima


def save_maps(xmmsim, outname, countmap, expmap, bkgmap, write_arf=False):
    """
    Function to save generated maps into output files

    :param xmmsim: XMMSimulator
    :param outname: Name of output files
    :param countmap: Count map
    :param expmap: Exposure map
    :param bkgmap: QPB map
    """

    # Get mask file
    mask_file = get_data_file_path('imgs/%s_mask.fits.gz' % (xmmsim.instrument))

    inmask = fits.open(mask_file)

    head_mask = inmask[1].header

    mask = inmask[1].data

    npix = mask.shape[0]

    inmask.close()

    hdu = fits.PrimaryHDU(countmap.astype(np.int16))

    header = hdu.header

    header['CDELT1'] = head_mask['CDELT1']
    header['CDELT2'] = head_mask['CDELT2']
    header['CTYPE1'] = head_mask['CTYPE1']
    header['CTYPE2'] = head_mask['CTYPE2']
    header['CRPIX1'] = head_mask['CRPIX1']
    header['CRPIX2'] = head_mask['CRPIX2']
    header['CRVAL1'] = 0.
    header['CRVAL2'] = 0.
    today = datetime.date(datetime.now())
    header['DATE'] = today.isoformat()
    header['CREATOR'] = 'xmm_simulator'
    header['TELESCOP'] = 'XMM'
    header['INSTRUME'] = 'E'+xmmsim.instrument
    header['CONTENT'] = 'COUNT MAP'
    header['DURATION'] = np.max(expmap)

    hdu.header = header

    hdu.writeto(outname+'.fits', overwrite=True)

    hdu.data = expmap.astype(np.float32)

    header['CONTENT'] = 'EXPOSURE MAP'

    hdu.header = header

    hdu.writeto(outname+'_expo.fits', overwrite=True)

    hdu.data = bkgmap

    header['CONTENT'] = 'QPB MAP'

    hdu.header = header

    hdu.writeto(outname + '_qpb.fits', overwrite=True)

    if write_arf:
        # Write ARF
        rmf_file = get_data_file_path('rmfs/%s.rmf' % (xmmsim.instrument))
        rmf = OGIPResponse(rsp_file=rmf_file)

        nchan = len(rmf.monte_carlo_energies) - 1
        mc_ene_lo = rmf.monte_carlo_energies[:nchan]
        mc_ene_hi = rmf.monte_carlo_energies[1:]

        arf_onaxis = calc_arf(theta=0.0, ebound_lo=mc_ene_lo, ebound_hi=mc_ene_hi, xmmsim=xmmsim)

        hdul = fits.HDUList([fits.PrimaryHDU()])
        cols = []
        cols.append(fits.Column(name='ENERG_LO', format='J', unit='keV', array=mc_ene_lo))
        cols.append(fits.Column(name='ENERG_HI', format='J', unit='keV', array=mc_ene_hi))
        cols.append(fits.Column(name='SPECRESP', format='J', unit='cm2', array=arf_onaxis))
        cols = fits.ColDefs(cols)
        tbhdu = fits.BinTableHDU.from_columns(cols, name='SPECRESP')
        hdr = tbhdu.header
        hdr['ARFVERSN'] = '1992a'
        hdr['HDUCLASS'] = 'OGIP'
        hdr['HDUCLAS1'] = 'RESPONSE'
        hdr['HDUCLAS2'] = 'SPECRESP'
        hdr['HDUVERS1'] = '1.3.0'
        hdr['ORIGIN'] = 'UNIGE'
        hdr['CREATOR'] = 'xmm_simulator'
        hdr['TELESCOP'] = 'XMM'
        hdr['INSTRUME'] = 'E'+xmmsim.instrument
        hdr['OBS_MODE'] = 'FullFrame'
        hdr['FILTER'] = 'Medium'
        hdr['DATE'] = today.isoformat()
        hdul.append(tbhdu)

        hdul.writeto(outname+'.arf', overwrite=True)



def sum_maps(dir, maps, instruments, pnfact):
    """
    Sum maps present within a directory to create a summed EPIC map with weighted exposure map

    :param dir:
    :param maps:
    :param instruments:
    :param pnfact:
    """
    cwd = os.getcwd()

    os.chdir(dir)

    nmaps = len(maps)
    print('Summing up %d maps...' % (nmaps))

    ima, expo, qpb = None, None, None
    fin, fexp, fqpb = None, None, None

    if nmaps>0:
        name = maps[0].split('.')[0]

        fin = fits.open(maps[0])

        ima = np.copy(fin[0].data)

        fexp = fits.open(name+'_expo.fits')

        if 'MOS' in instruments[0]:
            expo = fexp[0].data
        else:
            expo = fexp[0].data * pnfact

        fqpb = fits.open(name+'_qpb.fits')

        qpb = fqpb[0].data

    for i in range(1,nmaps):
        name = maps[i].split('.')[0]

        fint = fits.open(maps[i])

        ima = ima + np.copy(fint[0].data)

        fexpt = fits.open(name + '_expo.fits')

        if 'MOS' in instruments[i]:
            expo = expo + fexpt[0].data
        else:
            expo = expo + fexpt[0].data * pnfact

        fqpbt = fits.open(name + '_qpb.fits')

        qpb = qpb + fqpbt[0].data

        fint.close()
        fexpt.close()
        fqpbt.close()

    if nmaps>0:
        fin[0].data = ima
        fin.writeto('epic.fits', overwrite=True)
        fin.close()

        fexp[0].data = expo
        fexp.writeto('epic_expo.fits', overwrite=True)
        fexp.close()

        fqpb[0].data = qpb
        fqpb.writeto('epic_qpb.fits', overwrite=True)
        fqpb.close()

    os.chdir(cwd)




