import numpy as np
from astropy.io import fits
from .utils import get_data_file_path
from .imaging import psf_convole
from .background import tot_area
from scipy.interpolate import interp1d, RectBivariateSpline
from threeML.utils.OGIP.response import OGIPResponse
from threeML import APEC, Powerlaw, PhAbs
import progressbar
from datetime import datetime

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

    pixsize_ori = xmmsim.box_size / xmmsim.box.shape[1] # arcmin

    cx, cy = npix_out / 2., npix_out / 2.

    skybkg_spectrum = None

    if with_skybkg:
        modlhb = APEC()
        modgh = APEC()
        modcxb = Powerlaw()

        #modlhb.init_session()
        #modgh.init_session()

        modlhb.kT = 0.11
        modlhb.K = lhb
        modlhb.redshift = 0.
        modgh.kT = ght
        modgh.K = ghn
        modgh.redshift = 0.
        modcxb.index = -1.46
        modcxb.K = cxb
        modphabs = PhAbs()
        #modphabs.init_xsect()

        modphabs.NH = NH

        modsource = pixsize_ori**2 * (modlhb + modphabs * (modgh + modcxb))

        skybkg_spectrum = modsource(xmmsim.box_ene_mean)

    # Get photons per channel
    phot_box = xmmsim.box * tsim * xmmsim.all_arfs

    if xmmsim.pts:
        phot_box = phot_box + xmmsim.box_pts

    if with_skybkg:
        for i in range(len(xmmsim.box_ene_mean)):

            phot_box[:, :, i] = phot_box[:, :, i] + skybkg_spectrum[i] * tsim * xmmsim.all_arfs[:, :, i]

    nene = len(ene) - 1

    xnew = (np.arange(0, npix_out, 1) - cx) * pixsize / pixsize_ori

    ynew = (np.arange(0, npix_out, 1) - cy) * pixsize / pixsize_ori

    phot_box_ima = np.empty((npix_out,npix_out,nene))

    bar = progressbar.ProgressBar()

    for i in bar(range(nene)):
        # Select box data in the chosen energy band
        ima = phot_box[:, :, i]

        # Recast box shape into output image shape
        cx_ori, cy_ori =  xmmsim.box.shape[1]/2. , xmmsim.box.shape[0]/2.

        finterp = RectBivariateSpline(yori - cy_ori, xori - cx_ori, ima.T)

        ima_newpix = finterp(xnew, ynew).T * (pixsize / pixsize_ori)**2 # phot/cm2/s/keV

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

    bar = progressbar.ProgressBar()

    for i in bar(range(npix_active)):
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


def gen_image_evt(xmmsim, X_evt, Y_evt, chan_evt, tsim, elow=0.5, ehigh=2.0, outfile=None):
    # Get mask file
    mask_file = get_data_file_path('imgs/%s_mask.fits.gz' % (xmmsim.instrument))

    inmask = fits.open(mask_file)

    mask = inmask[1].data

    npix_mask = mask.shape[0]

    head_mask = inmask[1].header

    inmask.close()

    eband = np.where(np.logical_and(chan_evt >= elow, chan_evt <= ehigh))

    bin_ima = np.arange(0, npix_mask+1, 1)

    (ima, bin_x, bin_y) = np.histogram2d(Y_evt[eband], X_evt[eband], bins=bin_ima)

    if outfile is not None:
        hdu = fits.PrimaryHDU(ima.astype(np.int16))

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
        header['INSTRUME'] = 'E' + xmmsim.instrument
        header['CONTENT'] = 'COUNT MAP'
        header['DURATION'] = tsim

        hdu.header = header

        hdu.writeto(outfile, overwrite=True)

    return ima


def gen_qpb_evt(xmmsim, tsim):
    # Get mask file
    mask_file = get_data_file_path('imgs/%s_mask.fits.gz' % (xmmsim.instrument))

    inmask = fits.open(mask_file)

    mask = inmask[1].data

    pixsize = inmask[1].header['CDELT2'] * 60.  # arcmin

    inmask.close()

    # Set area to the area of active pixels
    area_tot = tot_area(xmmsim)

    area_mask = np.sum(mask) * pixsize ** 2

    fwc_m1 = xmmsim.fwc_spec * area_mask / area_tot

    # Generate events
    emin, emax = xmmsim.ebounds['E_MIN'], xmmsim.ebounds['E_MAX']

    mc_ene = (emax + emin) / 2.

    bin_width = emax - emin

    qpb_spec = np.random.poisson(fwc_m1 * tsim * bin_width).astype(int)

    evts = np.repeat(mc_ene, qpb_spec)

    # Place events on the active pixels
    y_mask, x_mask = np.indices(mask.shape)

    nonz = np.where(mask > 0)

    y_active, x_active = y_mask[nonz], x_mask[nonz]

    num_active = len(x_active)

    ind = range(num_active)

    sel = np.random.choice(ind, size=len(evts))

    x_sel = x_active[sel]

    y_sel = y_active[sel]

    return x_sel, y_sel, evts

def merge_evt(all_x, all_y, all_chan, tsim):
    '''
    Merge events from various lists (source, sky bkg, qpb) and randomize time of arrival

    :param all_x:
    :param all_y:
    :param all_chan:
    :param tsim:
    :return:
    '''

    nmerg = len(all_x)

    time_evt = np.array([])

    x_tot, y_tot, chan_tot = np.array([]), np.array([]), np.array([])

    for i in range(nmerg):

        X_evt = all_x[i]

        nevt = len(X_evt)

        itime = np.random.rand(nevt) * tsim

        time_evt = np.append(time_evt, itime)

        x_tot = np.append(x_tot, X_evt)

        y_tot = np.append(y_tot, all_y[i])

        chan_tot = np.append(chan_tot, all_chan[i])

    args = np.argsort(time_evt)

    return x_tot[args].astype(int), y_tot[args].astype(int), chan_tot[args], time_evt[args]

def save_evt_file(xmmsim, X_evt, Y_evt, chan_evt, time_evt, tsim, outfile):
    '''

    :param xmmsim:
    :param X_evt:
    :param Y_evt:
    :param chan_evt:
    :param outfile:
    :return:
    '''

    hdul = fits.HDUList([fits.PrimaryHDU()])

    cols = []
    cols.append(fits.Column(name='TIME', format='1E', array=time_evt))
    cols.append(fits.Column(name='X', format='1E', array=X_evt))
    cols.append(fits.Column(name='Y', format='1E', array=Y_evt))
    cols.append(fits.Column(name='ENERGY', format='1E', unit='keV', array=chan_evt))
    cols = fits.ColDefs(cols)
    tbhdu = fits.BinTableHDU.from_columns(cols, name='EVENTS')
    hdr = tbhdu.header
    hdr['ORIGIN'] = 'UNIGE'
    hdr['CREATOR'] = 'xmm_simulator'
    hdr['TELESCOP'] = ('XMM', 'Telescope (mission) name')
    hdr['INSTRUME'] = ('E'+xmmsim.instrument, 'Instrument name')
    hdr['OBS_MODE'] = 'FullFrame'
    hdr['FILTER'] = ('Medium', 'Instrument filter in use')
    today = datetime.date(datetime.now())
    hdr['DATE'] = today.isoformat()
    hdr['RA_OBJ'] = 0.0
    hdr['DEC_OBJ'] = 0.0
    hdr['DATE-OBS'] = today.isoformat()
    hdr['ONTIME'] = tsim
    hdr['EXPOSURE'] = (tsim, 'Weighted live time of CCDs in the extraction region')
    hdul.append(tbhdu)

    hdul.writeto(outfile, overwrite=True)
    print('Event file written to file', outfile)

    hdul.close()

