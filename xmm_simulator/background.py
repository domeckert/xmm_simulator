from astropy.io import fits
import numpy as np
from .utils import get_data_file_path, calc_arf, region
from threeML import APEC, Powerlaw, PhAbs
from threeML.utils.OGIP.response import OGIPResponse
from scipy.ndimage import gaussian_filter1d

area_in_pn = 610.9
area_out_pn = 52.2505

area_in_m2 = 618.7
area_out_m2 = area_in_m2 / 3.261834

lhb_ref = 2.92859e-06 # MACS 0949 sky bkg parameters
ght_ref = 0.220899
ghn_ref = 5.02297e-07
cxb_ref = 7.94099e-07
NH_ref = 0.05

def tot_area(xmmsim):
    if xmmsim.instrument == 'PN':
        area_tot = area_in_pn + area_out_pn

    else:
        area_tot = area_in_m2 + area_out_m2

    return area_tot

def gen_qpb_spectrum(xmmsim):
    """
    Generate a filter-wheel-closed spectrum for a given observation time and source area

    :param xmmsim: XMMSimulator
    :param tsim: Simulation exposure time in second
    :param area_spec: Source area in square arcmin
    :return: spec_bkg: Background QPB spectrum
    """

    rmf_file = get_data_file_path('rmfs/%s.rmf' % (xmmsim.instrument))

    frmf = fits.open(rmf_file)
    ebounds = frmf['EBOUNDS'].data
    frmf.close()

    fwc_file = fits.open(xmmsim.ccfpath + xmmsim.fwc_file)

    evts_fwc = fwc_file[1].data
    exp_fwc = fwc_file[2].data

    okflag = np.where(evts_fwc['FLAG'] == 0)

    nevt_tot = len(okflag[0])

    if xmmsim.instrument == 'PN':

        sum_expo = np.sum(exp_fwc['EXPOSURE']) / 12. # PN has 12 chips

    else:

        sum_expo = np.sum(exp_fwc['EXPOSURE']) / 7. # MOS has 7 chips

    emin, emax = ebounds['E_MIN'], ebounds['E_MAX']

    nchan = len(emin)

    spec_rate, bin_width = np.empty(nchan), np.empty(nchan)

    evt_okflag = evts_fwc[okflag]

    for i in range(nchan):
        sel_chan = np.where(np.logical_and(evt_okflag['PI'] >= emin[i] * 1000, evt_okflag['PI'] < emax[i] * 1000))

        bin_width[i] = emax[i] - emin[i]

        spec_rate[i] = len(sel_chan[0]) / sum_expo / bin_width[i] # cts/s/keV

    spec_rate_smoothed = gaussian_filter1d(spec_rate, 5.)

    return spec_rate_smoothed, ebounds


def gen_qpb_image(xmmsim, tsim, elow=0.5, ehigh=2.0):
    """

    :param xmmsim:XMMSimulator
    :param tsim: Simulation exposure time in second
    :param elow: Lower energy boundary of the image
    :param ehigh: Upper energy boundary of the image
    :return: bkg_map: QPB background map
    """

    mask_file = get_data_file_path('imgs/%s_mask.fits.gz' % (xmmsim.instrument))

    inmask = fits.open(mask_file)

    mask = inmask[1].data

    pixsize = inmask[1].header['CDELT2'] * 60.  # arcmin

    fwc_file = fits.open(xmmsim.ccfpath + xmmsim.fwc_file)

    evts_fwc = fwc_file[1].data
    exp_fwc = fwc_file[2].data

    okflag = np.where(evts_fwc['FLAG'] == 0)

    if xmmsim.instrument == 'PN':
        area_tot = area_in_pn + area_out_pn

        sum_expo = np.sum(exp_fwc['EXPOSURE']) / 12. # PN has 12 chips

    else:
        area_tot = area_in_m2 + area_out_m2

        sum_expo = np.sum(exp_fwc['EXPOSURE']) / 7. # MOS has 7 chips

    evt_okflag = evts_fwc[okflag]

    sel_phot_ene = np.where(np.logical_and(evt_okflag['PI'] > elow * 1000, evt_okflag['PI'] < ehigh * 1000))

    rate_phot = len(sel_phot_ene[0]) / sum_expo / area_tot

    bkg_map = mask * rate_phot * pixsize ** 2 * tsim

    fwc_file.close()

    inmask.close()

    return bkg_map

def gen_skybkg_spectrum(xmmsim, tsim, area_spec, arf, lhb=None, ght=None, ghn=None, cxb=None, NH=None):
    """
    Generate a sky background spectrum for a given observation source area

    :param area_spec: Source area in square arcmin
    :return: spec_bkg: Sky background photon spectrum
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

    modsource = area_spec * (modlhb + modphabs * (modgh + modcxb))

    # Read RMF
    rmf_file = get_data_file_path('rmfs/%s.rmf' % (xmmsim.instrument))

    rmf = OGIPResponse(rsp_file=rmf_file)

    nchan = len(rmf.monte_carlo_energies) - 1

    mc_ene = (rmf.monte_carlo_energies[:nchan] + rmf.monte_carlo_energies[1:]) / 2.

    # Compute model
    spec_phot = modsource(mc_ene) * tsim * arf

    # Convolve with RMF
    bin_width = rmf.monte_carlo_energies[1:] - rmf.monte_carlo_energies[:nchan]

    spec_conv = rmf.convolve(spec_phot * bin_width)

    modlhb.clean()
    modgh.clean()

    return spec_conv


def gen_skybkg_image(xmmsim, tsim, elow=0.5, ehigh=2.0, nbin=10, lhb=None, ght=None, ghn=None, cxb=None, NH=None):
    """

    :param xmmsim:XMMSimulator
    :param tsim: Simulation exposure time in second
    :param elow: Lower energy boundary of the image
    :param ehigh: Upper energy boundary of the image
    :param nbin: Number of sub-bins inside which the vignetting curve will be computed
    :return:
        - bkg_map: Sky background map
        - exp_map: Exposure map
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

    # Set up sky background model
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

    expmap, bkg_map = np.zeros(mask.shape), np.zeros(mask.shape)

    cx, cy = mask.shape[0] / 2., mask.shape[1] / 2.

    y, x = np.indices(mask.shape)

    thetas = np.hypot(x - cx, y - cy) * pixsize  # arcmin

    arf_onaxis = calc_arf(0., ene_bins[:nbin], ene_bins[1:], xmmsim)

    skybkg_spectrum = modsource(ene_bins)

    for i in range(nbin):

        ene_bin = (ene_bins[i] + ene_bins[i+1]) / 2.

        nearest = np.argsort(np.abs(ene_vig - ene_bin))[0]

        vig_curve = vig_fact[nearest]

        texp = np.interp(thetas, rads, vig_curve) * tsim

        expmap = expmap + texp

        tsky = (skybkg_spectrum[i] + skybkg_spectrum[i + 1]) / 2.

        bkg_map = bkg_map + texp * arf_onaxis[i] * tsky * ene_bin_width

    expmap = expmap / nbin

    expmap = expmap * mask

    bkg_map = bkg_map * mask

    modlhb.clean()
    modgh.clean()

    return  bkg_map, expmap

