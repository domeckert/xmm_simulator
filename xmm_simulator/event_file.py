import numpy as np
from astropy.io import fits
from .utils import get_data_file_path
from .imaging import psf_convole
from scipy.interpolate import interp2d, interp1d
import os
from threeML.utils.OGIP.response import OGIPResponse
from threeML import APEC, Powerlaw, PhAbs
import progressbar

lhb_ref = 2.92859e-06 # MACS 0949 sky bkg parameters
ght_ref = 0.220899
ghn_ref = 5.02297e-07
cxb_ref = 7.94099e-07
NH_ref = 0.05

def gen_phot_box(xmmsim, tsim, with_skybkg=True, lhb=None, ght=None, ghn=None, cxb=None, NH=None):
    """
    Generate a box expectation value in photon/keV , convolved with the PSF and multiplied by the detector mask

    :param xmmsim: XMMSimulator
    :param tsim: Simulation exposure time in sec
    :return: photon box
    """

    if lhb is None:
        lhb = lhb_ref

    if ght is None:
        ght = ght_ref

    if ghn is None:
        ghn = ghn_ref

    if cxb is None:
        cxb = cxb_ref

    if NH is None:
        NH = NH_ref


    if xmmsim.all_arfs is None:
        print('ARF cube not found, please compute ARFs first')

        return

    # Get mask file
    mask_file = get_data_file_path('imgs/%s_mask.fits.gz' % (xmmsim.instrument))

    inmask = fits.open(mask_file)

    mask = inmask[1].data

    pixsize = inmask[1].header['CDELT2'] * 60.  # arcmin

    inmask.close()

    npix_out = mask.shape[0]

    ene = xmmsim.box_ene

    xori = np.arange(0, xmmsim.box.shape[1], 1)
    yori = np.arange(0, xmmsim.box.shape[0], 1)

    pixsize_ori = 30. / xmmsim.box.shape[1] # arcmin

    cx, cy = npix_out / 2., npix_out / 2.

    skybkg_spectrum = None

    if with_skybkg:
        modlhb = APEC()
        modgh = APEC()
        modcxb = Powerlaw()

        modlhb.init_session()
        modgh.init_session()

        modlhb.kT = 0.11
        modlhb.K = lhb
        modlhb.redshift = 0.
        modgh.kT = ght
        modgh.K = ghn
        modgh.redshift = 0.
        modcxb.index = -1.46
        modcxb.K = cxb
        modphabs = PhAbs()
        modphabs.init_xsect()

        modphabs.NH = NH

        modsource = pixsize**2 * (modlhb + modphabs * (modgh + modcxb))

        skybkg_spectrum = modsource(xmmsim.box_ene_mean)

    # Get photons per channel
    phot_box = xmmsim.box * tsim * xmmsim.all_arfs

    if with_skybkg:
        for i in range(len(xmmsim.box_ene_mean)):

            phot_box[:, :, i] = phot_box[:, :, i] + skybkg_spectrum[i] * tsim * xmmsim.all_arfs[:, :, i]

    nene = len(ene) - 1

    xnew = (np.arange(0, npix_out, 1) - cx) * pixsize / pixsize_ori

    ynew = (np.arange(0, npix_out, 1) - cy) * pixsize / pixsize_ori

    phot_box_ima = np.empty((npix_out,npix_out,nene))

    for i in progressbar.progressbar(range(nene)):
        # Select box data in the chosen energy band
        ima = phot_box[:, :, i]

        # Recast box shape into output image shape
        cx_ori, cy_ori =  xmmsim.box.shape[1]/2. , xmmsim.box.shape[0]/2.

        finterp = interp2d(yori - cy_ori, xori - cx_ori, ima, kind='linear')

        ima_newpix = finterp(xnew, ynew) * (pixsize / pixsize_ori)**2 # phot/cm2/s/keV

        phot_box_ima[:,:,i] = psf_convole(ima_newpix, pixsize, xmmsim) * mask

    return phot_box_ima


# test by simulating the number of photons first and then drawing the energy
def gen_evt_list(xmmsim, phot_box_ima):
    # Read RMF
    rmf_file = get_data_file_path('rmfs/%s.rmf' % (xmmsim.instrument))

    rmf = OGIPResponse(rsp_file=rmf_file)

    nchan = len(rmf.monte_carlo_energies) - 1

    mc_ene = (rmf.monte_carlo_energies[:nchan] + rmf.monte_carlo_energies[1:]) / 2.

    bin_width = rmf.monte_carlo_energies[1:] - rmf.monte_carlo_energies[:nchan]

    ima_tot = np.sum(phot_box_ima, axis=2) * xmmsim.box_ene_width

    photon_map = np.random.poisson(ima_tot)

    yp, xp = np.indices(photon_map.shape)

    nonz = np.where(photon_map > 0)

    X_evt = np.repeat(xp[nonz], photon_map[nonz])

    Y_evt = np.repeat(yp[nonz], photon_map[nonz])

    chan_evt = np.array([])

    npix_active = len(xp[nonz])

    for i in progressbar.progressbar(range(npix_active)):
        tx = xp[nonz][i]
        ty = yp[nonz][i]

        nphot_pix = photon_map[nonz][i]

        phot_spec = phot_box_ima[ty, tx, :]

        # Interpolate to the proper channels
        finterp_spec = interp1d(xmmsim.box_ene_mean, phot_spec, fill_value='extrapolate')

        spec_interp = finterp_spec(mc_ene)

        wronginterp = np.where(spec_interp < 0.)

        spec_interp[wronginterp] = 0.

        # Convolve with RMF
        spec_conv = rmf.convolve(spec_interp * bin_width)

        spec_prob = spec_conv / np.sum(spec_conv)

        evts = np.random.choice(mc_ene, p=spec_prob, size=nphot_pix)

        chan_evt = np.append(chan_evt, evts)

    return X_evt, Y_evt, chan_evt


def gen_image_evt(xmmsim, X_evt, Y_evt, chan_evt, emin=0.5, emax=2.0, outfile=None):
    # Get mask file
    mask_file = get_data_file_path('imgs/%s_mask.fits.gz' % (xmmsim.instrument))

    inmask = fits.open(mask_file)

    mask = inmask[1].data

    npix_mask = mask.shape[0]

    eband = np.where(np.logical_and(chan_evt >= emin, chan_evt <= emax))

    bin_ima = np.arange(0, npix_mask, 1)

    (ima, bin_x, bin_y) = np.histogram2d(Y_evt[eband], X_evt[eband], bins=bin_ima)

    if outfile is not None:
        inmask[1].data = ima

        inmask.writeto(outfile, overwrite=True)

    inmask.close()

    return ima

def save_evt_file(xmmsim, X_evt, Y_evt, chan_evt, outfile):
    '''

    :param xmmsim:
    :param X_evt:
    :param Y_evt:
    :param chan_evt:
    :param outfile:
    :return:
    '''

