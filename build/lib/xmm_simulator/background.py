from astropy.io import fits
import numpy as np
from .utils import get_data_file_path
from threeML import APEC, Powerlaw, PhAbs
from scipy.interpolate import interp2d

area_in_pn = 610.9
area_out_pn = 52.2505

area_in_m2 = 618.7
area_out_m2 = area_in_m2 / 3.261834

lhb_ref = 2.73025e-06 # MACS 0949 sky bkg parameters
ght_ref = 0.261624
ghn_ref = 5.90513e-07
cxb_ref = 1.07595e-06
NH_ref = 0.0295

def gen_qpb_spectrum(xmmsim, tsim, area_spec):
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
        area_tot = area_in_pn + area_out_pn

        sum_expo = np.sum(exp_fwc['EXPOSURE']) / 12. # PN has 12 chips

    else:
        area_tot = area_in_m2 + area_out_m2

        sum_expo = np.sum(exp_fwc['EXPOSURE']) / 7. # MOS has 7 chips

    nevt_rat = nevt_tot * tsim / sum_expo * area_spec / area_tot

    nevt_sim = np.random.poisson(nevt_rat)

    rand_evt = np.random.rand(nevt_sim) * nevt_sim

    sel_evt = (rand_evt.astype(int))

    evt_okflag = evts_fwc[okflag]

    evtlist_sel = evt_okflag[sel_evt]

    emin, emax = ebounds['E_MIN'], ebounds['E_MAX']

    nchan = len(emin)

    spec_bkg = np.empty(nchan)

    for i in range(nchan):
        sel_chan = np.where(np.logical_and(evtlist_sel['PI'] >= emin[i] * 1000, evtlist_sel['PI'] < emax[i] * 1000))

        spec_bkg[i] = len(sel_chan[0])

    spec_bkg = spec_bkg.astype(int)

    return spec_bkg


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

    mask_file.close()

    return bkg_map

def gen_skybkg_spectrum(area_spec, chan_lo, chan_hi, lhb=None, ght=None, ghn=None, cxb=None, NH=None):
    """
    Generate a sky background spectrum for a given observation source area

    :param area_spec: Source area in square arcmin
    :param chan_lo: Lower energy boundaries of the energy channels
    :param chan_hi: Upper energy boundaries of the energy channels
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

    chan_mean = (chan_lo + chan_hi) / 2.

    spec_bkg = modsource(chan_mean)

    return spec_bkg

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

    # Reading vignetting curve
    dtheta = xmmsim.dtheta

    rads = np.arange(0., 16., dtheta)

    ene_vig = xmmsim.vignetting['ENERGY'] / 1e3
    vig_fact = xmmsim.vignetting['VIGNETTING_FACTOR']

    ene_bins = np.linspace(elow, ehigh, nbin+1)

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

    expmap = np.copy(mask)

    cx, cy = mask.shape[0] / 2., mask.shape[1] / 2.

    y, x = np.indices(mask.shape)

    thetas = np.hypot(x - cx, y - cy) * pixsize  # arcmin

    for i in range(nbin):

        ene_bin = (ene_bins[i] + ene_bins[i+1]) / 2.

        fvig_interp = interp2d(rads, ene_vig, vig_fact, kind='cubic')

        vig_theta = fvig_interp(thetas, ene_vig).flatten()

