import numpy as np
from astropy.io import fits
from .utils import region, set_wcs, get_data_file_path, region_evt
from threeML.utils.OGIP.response import OGIPResponse
from scipy.interpolate import interp1d, RectBivariateSpline
from datetime import datetime
import os

def gen_spec_box(xmmsim, tsim, cra, cdec, rin, rout, regfile=None):
    """
    Generate a predicted spectrum from a box within an annulus

    :param xmmsim:
    :param tsim:
    :param cra:
    :param cdec:
    :param rin:
    :param rout:
    :param region:
    :return:
        - Spectrum
        - ARF
        - Backscale
    """

    if xmmsim.all_arfs is None:
        print('ARF cube not found, please compute ARFs first')

        return

    box = xmmsim.box

    pixsize = xmmsim.box_size / box.shape[0]  # arcmin

    # Set region definition
    y, x = np.indices(box[:, :, 0].shape)

    wcs = set_wcs(xmmsim=xmmsim, type='box')

    wc = np.array([[cra, cdec]])

    pixcrd = wcs.wcs_world2pix(wc, 1)

    xsrc = pixcrd[0][0] - 1.

    ysrc = pixcrd[0][1] - 1.

    thetas = np.hypot(x - xsrc, y - ysrc) * pixsize  # arcmin

    if regfile is not None:

        thetas = region(regfile=regfile,
                        thetas=thetas,
                        wcs_inp=wcs,
                        pixsize=pixsize)

    test_annulus = np.where(np.logical_and(thetas >= rin, thetas < rout))

    arfs_sel = xmmsim.all_arfs[test_annulus]

    box_sel = box[test_annulus]

    # Get photons per channel
    phot_box = box_sel * tsim * arfs_sel

    phot_spec = np.sum(phot_box, axis=0)

    # Read RMF
    rmf_file = get_data_file_path('rmfs/%s.rmf' % (xmmsim.instrument))

    rmf = OGIPResponse(rsp_file=rmf_file)

    nchan = len(rmf.monte_carlo_energies) - 1

    mc_ene = (rmf.monte_carlo_energies[:nchan] + rmf.monte_carlo_energies[1:]) / 2.

    # Interpolate to the proper channels
    finterp_spec = interp1d(xmmsim.box_ene_mean, phot_spec, fill_value='extrapolate')

    spec_interp = finterp_spec(mc_ene)

    wronginterp = np.where(spec_interp < 0.)

    spec_interp[wronginterp] = 0.

    # Convolve with RMF
    bin_width = rmf.monte_carlo_energies[1:] - rmf.monte_carlo_energies[:nchan]

    spec_conv = rmf.convolve(spec_interp * bin_width)

    # Compute mean ARF and interpolate
    arf_mean = np.average(arfs_sel, axis=0, weights=box_sel)

    finterp_arf = interp1d(xmmsim.box_ene_mean, arf_mean, fill_value='extrapolate')

    arf_mean_interp = finterp_arf(mc_ene)

    neg_arf = np.where(arf_mean_interp < 0.)

    arf_mean_interp[neg_arf] = 0.

    # Compute BACKSCAL
    backscal_annulus = box_sel.shape[0] * (pixsize * 60.) ** 2 / (0.05 ** 2)

    return spec_conv, arf_mean_interp, backscal_annulus



def gen_spec_evt(xmmsim, cra, cdec, rin, rout, regfile=None):
    """
    Generate a predicted spectrum from a box within an annulus

    :param xmmsim:
    :param tsim:
    :param cra:
    :param cdec:
    :param rin:
    :param rout:
    :param region:
    :return:
        - Spectrum
        - ARF
        - Backscale
    """

    if not xmmsim.events:
        print('Event file not extracted yet, aborting')
        return

    # Get mask file
    mask_file = get_data_file_path('imgs/%s_mask.fits.gz' % (xmmsim.instrument))

    inmask = fits.open(mask_file)

    mask = inmask[1].data

    pixsize = inmask[1].header['CDELT2'] * 60.  # arcmin

    pixsize_ori = xmmsim.box_size / xmmsim.box.shape[0]  # arcmin

    npix_out = mask.shape[0]

    inmask.close()

    # Set region definition
    y, x = np.indices(xmmsim.box[:, :, 0].shape)

    wcs_mask = set_wcs(xmmsim=xmmsim, type='mask')

    wcs_box = set_wcs(xmmsim=xmmsim, type='box')

    wc = np.array([[cra, cdec]])

    pixcrd = wcs_mask.wcs_world2pix(wc, 1)

    xsrc = pixcrd[0][0] - 1.

    ysrc = pixcrd[0][1] - 1.

    pixcrd_box = wcs_box.wcs_world2pix(wc, 1)

    xsrc_box = pixcrd_box[0][0] - 1.

    ysrc_box = pixcrd_box[0][1] - 1.

    thetas = np.hypot(xmmsim.X_evt - xsrc, xmmsim.Y_evt - ysrc) * pixsize  # arcmin

    thetas_ima = np.hypot(x - xsrc_box, y - ysrc_box) * pixsize_ori

    # Recast mask shape into box image shape
    cx, cy = npix_out / 2., npix_out / 2.
    cx_box, cy_box = xmmsim.box.shape[1] / 2., xmmsim.box.shape[0] / 2.
    xmask = (np.arange(0, npix_out, 1) - cx) * pixsize / pixsize_ori
    ymask = (np.arange(0, npix_out, 1) - cy) * pixsize / pixsize_ori
    xbox = np.arange(0, xmmsim.box.shape[1], 1) - cx_box
    ybox = np.arange(0, xmmsim.box.shape[0], 1) - cy_box

    finterp = RectBivariateSpline(ymask, xmask, mask)

    # mask unobserved areas
    mask_ori = np.floor(finterp(xbox, ybox) + 0.5).astype(int)

    tbm = np.where(mask_ori == 0)
    thetas_ima[tbm] = -1

    if regfile is not None:
        thetas = region_evt(xmmsim=xmmsim,
                            regfile=regfile,
                            thetas=thetas,
                            wcs_inp=wcs_mask,
                            pixsize=pixsize)

        thetas_ima = region(regfile=regfile,
                            thetas=thetas_ima,
                            wcs_inp=wcs_box,
                            pixsize=pixsize_ori)

    test_annulus = np.where(np.logical_and(thetas >= rin, thetas < rout))

    # Get photons per channel
    sel_phot = xmmsim.chan_evt[test_annulus]

    # Read RMF
    rmf_file = get_data_file_path('rmfs/%s.rmf' % (xmmsim.instrument))

    rmf = OGIPResponse(rsp_file=rmf_file)

    (spec_sel, bins_spec) = np.histogram(sel_phot, bins=rmf.ebounds)

    # Compute BACKSCAL
    sel_area = np.where(np.logical_and(thetas_ima >= rin, thetas_ima < rout))

    backscal_annulus = len(sel_area[0]) * (pixsize_ori * 60.) ** 2 / (0.05 ** 2)

    # Compute ARF
    arfs_sel = xmmsim.all_arfs[sel_area]

    box_sel = xmmsim.box[sel_area]

    if np.sum(box_sel) == 0:
        box_sel = np.ones(len(arfs_sel))

    arf_mean = np.average(arfs_sel, axis=0, weights=box_sel)

    finterp_arf = interp1d(xmmsim.box_ene_mean, arf_mean, fill_value='extrapolate')

    nchan = len(rmf.monte_carlo_energies) - 1

    mc_ene = (rmf.monte_carlo_energies[:nchan] + rmf.monte_carlo_energies[1:]) / 2.

    arf_mean_interp = finterp_arf(mc_ene)

    neg_arf = np.where(arf_mean_interp < 0.)

    arf_mean_interp[neg_arf] = 0.

    return spec_sel, arf_mean_interp, backscal_annulus


def gen_spec_evt_pix(xmmsim, pixlist):
    """
    Generate a predicted spectrum from image pixels in pixlist

    :param xmmsim:
    :param pixlist:
    :return:
        - Spectrum
        - ARF
        - Backscale
    """

    if not xmmsim.events:
        print('Event file not extracted yet, aborting')
        return

    # Get mask file
    mask_file = get_data_file_path('imgs/%s_mask.fits.gz' % (xmmsim.instrument))

    inmask = fits.open(mask_file)

    mask = inmask[1].data

    pixsize = inmask[1].header['CDELT2'] * 60.  # arcmin
    pixsize_ori = xmmsim.box_size / xmmsim.box.shape[0]  # arcmin

    #Pavement for the original box (e.g. 512x512)
    xx_origin, yy_origin = np.meshgrid(np.arange(xmmsim.box.shape[0]), np.arange(xmmsim.box.shape[1]))

    npix_out = mask.shape[0]
    inmask.close()

    # Set region definition
    wcs_mask = set_wcs(xmmsim=xmmsim, type='mask')

    wcs_box = set_wcs(xmmsim=xmmsim, type='box')


    # Recast mask shape into box image shape
    cx, cy = npix_out / 2., npix_out / 2.
    cx_box, cy_box = xmmsim.box.shape[1] / 2., xmmsim.box.shape[0] / 2.
    xmask = (np.arange(0, npix_out, 1) - cx) * pixsize / pixsize_ori
    ymask = (np.arange(0, npix_out, 1) - cy) * pixsize / pixsize_ori
    xbox = np.arange(0, xmmsim.box.shape[1], 1) - cx_box
    ybox = np.arange(0, xmmsim.box.shape[0], 1) - cy_box

    finterp = RectBivariateSpline(ymask, xmask, mask)

    # mask unobserved areas
    mask_ori = np.floor(finterp(xbox, ybox) + 0.5).astype(int)
    tbm = np.where(mask_ori == 0)
    xx_origin[tbm] = -1
    yy_origin[tbm] = -1

    evt_pix_tosel = np.vstack([xmmsim.Y_evt, xmmsim.X_evt]).T

    #select photons within pixlist
    pixels_set = set(map(tuple, pixlist))
    is_selected = np.array([tuple(photon) in pixels_set for photon in evt_pix_tosel])

    # Get photons per channel
    sel_phot = xmmsim.chan_evt[is_selected]

    # Read RMF
    rmf_file = get_data_file_path('rmfs/%s.rmf' % (xmmsim.instrument))

    rmf = OGIPResponse(rsp_file=rmf_file)

    (spec_sel, bins_spec) = np.histogram(sel_phot, bins=rmf.ebounds)

    # Compute BACKSCAL
    #find pixels in the original box hit by events
    sky_coords = wcs_mask.pixel_to_world(pixlist[:,1], pixlist[:,0])
    pixlist_orig_x, pixlist_orig_y = wcs_box.world_to_pixel(sky_coords)
    pixlist_orig_x = np.floor(pixlist_orig_x + 0.5).astype(int)
    pixlist_orig_y = np.floor(pixlist_orig_y + 0.5).astype(int)

    pixlist_orig = np.vstack([pixlist_orig_x, pixlist_orig_y]).T
    pixels_set_orig = set(map(tuple, pixlist_orig))

    sel_area_ = np.array(
        [tuple(pix) in pixels_set_orig for pix in np.vstack([xx_origin.ravel(), yy_origin.ravel()]).T]).reshape(
        xx_origin.shape)
    sel_area = np.where(sel_area_)
    backscal_pixels = len(sel_area[0]) * (pixsize_ori * 60.) ** 2 / (0.05 ** 2)

    # Compute ARF
    arfs_sel = xmmsim.all_arfs[sel_area]
    box_sel = xmmsim.box[sel_area]
    if np.sum(box_sel) == 0:
        box_sel = np.ones(len(arfs_sel))

    arf_mean = np.average(arfs_sel, axis=0, weights=box_sel)

    finterp_arf = interp1d(xmmsim.box_ene_mean, arf_mean, fill_value='extrapolate')

    nchan = len(rmf.monte_carlo_energies) - 1

    mc_ene = (rmf.monte_carlo_energies[:nchan] + rmf.monte_carlo_energies[1:]) / 2.

    arf_mean_interp = finterp_arf(mc_ene)

    neg_arf = np.where(arf_mean_interp < 0.)

    arf_mean_interp[neg_arf] = 0.

    return spec_sel, arf_mean_interp, backscal_pixels


def save_spectrum(xmmsim, outdir, spectrum, tsim, arf, qpb, backscal, tsim_qpb):
    """

    :param xmmsim:
    :param spectrum:
    :param arf:
    :param qpb:
    :param backscal:
    :param tsim_qpb:
    :return:
    """

    if not os.path.exists(outdir):

        os.mkdir(outdir)

    rmf_file = get_data_file_path('rmfs/%s.rmf' % (xmmsim.instrument))

    rmf = OGIPResponse(rsp_file=rmf_file)

    nchan = len(rmf.monte_carlo_energies) - 1

    mc_ene = (rmf.monte_carlo_energies[:nchan] + rmf.monte_carlo_energies[1:]) / 2.

    pref = None
    if xmmsim.instrument == 'MOS1':
        pref = 'mos1S001'

    elif xmmsim.instrument == 'MOS2':
        pref = 'mos2S002'

    elif xmmsim.instrument == 'PN':
        pref = 'pnS003'

    nam = None
    if '/' not in outdir:
        nam = outdir
    else:
        tl = outdir.split('/')
        ntl = len(tl)
        nam = tl[ntl-1]

    name_spec = outdir + '/'+ pref + '-obj-'+ nam + '.pi'
    arf_name = outdir + '/'+ pref + '-' + nam + '.arf'
    bkg_name = outdir + '/'+ pref + '-back-'+ nam + '.pi'
    rmf_name = outdir + '/'+ pref + '-' + nam + '.rmf'

    os.system('cp %s %s' % (rmf_file, rmf_name))

    # Write spectrum
    channel = np.arange(0, len(spectrum), 1)

    hdul = fits.HDUList([fits.PrimaryHDU()])
    cols = []
    cols.append(fits.Column(name='CHANNEL', format='J', array=channel))
    cols.append(fits.Column(name='COUNTS', format='J', unit='count', array=spectrum))
    cols = fits.ColDefs(cols)
    tbhdu = fits.BinTableHDU.from_columns(cols, name='SPECTRUM')
    hdr = tbhdu.header
    hdr['HDUCLASS'] = ('OGIP', 'Format conforms to OGIP/GSFC conventions')
    hdr['HDUCLAS1'] = ('SPECTRUM', 'File contains a spectrum')
    hdr['HDUCLAS2'] = ('TOTAL', 'File contains gross counts')
    hdr['HDUCLAS3'] = ('COUNT', 'Spectrum is stored as counts')
    hdr['HDUVERS1'] = ('1.1.0', 'Version of format')
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
    hdr['CORRFILE'] = 'NONE'
    hdr['CORRSCAL'] = 1.
    hdr['POISSERR'] = (True, 'Poisson errors appropriate')
    hdr['QUALITY'] = 0
    hdr['GROUPING'] = 0
    hdr['RESPFILE'] = (rmf_name, 'redistribution matrix')
    hdr['ANCRFILE'] = (arf_name, 'ancillary response')
    hdr['BACKFILE'] = (bkg_name, 'Background FITS file')
    hdr['CHANTYPE'] = ('PI', 'Type of channel data')
    hdr['DETCHANS'] = len(channel)
    hdr['AREASCAL'] = (1., 'Nominal scaling factor for data')
    hdr['BACKSCAL'] = (backscal, 'Scaling factor for background')  # Sum of area in XMM units, 0.05 arcsec
    hdr['CTS'] = np.sum(spectrum).astype(int)
    hdul.append(tbhdu)

    hdul.writeto(name_spec, overwrite=True)
    print('Spectrum written to file', name_spec)

    hdul.close()

    if qpb is not None:
        # Write QPB
        hdul = fits.HDUList([fits.PrimaryHDU()])
        cols = []
        cols.append(fits.Column(name='CHANNEL', format='J', array=channel))
        cols.append(fits.Column(name='COUNTS', format='J', unit='count', array=qpb))
        cols = fits.ColDefs(cols)
        tbhdu = fits.BinTableHDU.from_columns(cols, name='SPECTRUM')
        hdr['ANCRFILE'] = 'none'
        hdr['BACKFILE'] = 'none'
        hdr['EXPOSURE'] = tsim_qpb
        hdr['ONTIME'] = tsim_qpb
        tbhdu.header = hdr
        hdul.append(tbhdu)

        hdul.writeto(bkg_name, overwrite=True)
        print('Background spectrum written to', bkg_name)

        hdul.close()

    if arf is not None:
        # Write ARF

        mc_ene_lo = rmf.monte_carlo_energies[:nchan]
        mc_ene_hi = rmf.monte_carlo_energies[1:]

        hdul = fits.HDUList([fits.PrimaryHDU()])
        cols = []
        cols.append(fits.Column(name='ENERG_LO', format='J', unit='keV', array=mc_ene_lo))
        cols.append(fits.Column(name='ENERG_HI', format='J', unit='keV', array=mc_ene_hi))
        cols.append(fits.Column(name='SPECRESP', format='J', unit='cm2', array=arf))
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

        hdul.writeto(arf_name, overwrite=True)
        print('ARF written to file', arf_name)

        hdul.close()


def save_spectrum_rassbkg(outdir, p2rass_rsp, spectrum, tsim, area_spec):
    """
    Function to save a mock generated rass background by the gen_skybkg_spectrum_rass in spectral.py
    The 'spectrum' passed to this function should be a possion realization of the model.
    :param outdir: Output directory to store the rass spectrum
    :param p2rass_rsp: Path to the rass response file
    :param spectrum: RASS spectrum to save
    :param tsim: Exposure time
    :param area_spec: Source area in square arcmin
    :return:
    """

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    rsp_file = p2rass_rsp


    pref = 'rass_spectrum'

    nam = None
    if '/' not in outdir:
        nam = outdir
    else:
        tl = outdir.split('/')
        ntl = len(tl)
        nam = tl[ntl - 1]

    name_spec = outdir + '/' + pref + '-obj-' + nam + '.pi'

    tsim_eff = tsim * area_spec

    # Write spectrum
    channel = np.arange(0, len(spectrum), 1)

    hdul = fits.HDUList([fits.PrimaryHDU()])
    cols = []
    cols.append(fits.Column(name='CHANNEL', format='J', array=channel))
    cols.append(fits.Column(name='COUNTS', format='J', unit='count', array=spectrum))
    cols = fits.ColDefs(cols)
    tbhdu = fits.BinTableHDU.from_columns(cols, name='SPECTRUM')
    hdr = tbhdu.header
    hdr['HDUCLASS'] = ('OGIP', 'Format conforms to OGIP/GSFC conventions')
    hdr['HDUCLAS1'] = ('SPECTRUM', 'File contains a spectrum')
    hdr['HDUCLAS2'] = ('TOTAL', 'File contains gross counts')
    hdr['HDUCLAS3'] = ('COUNT', 'Spectrum is stored as counts')
    hdr['HDUVERS1'] = ('1.1.0', 'Version of format')
    hdr['ORIGIN'] = 'UNIGE'
    hdr['CREATOR'] = 'xmm_simulator'
    hdr['TELESCOP'] = ('ROSAT', 'Telescope (mission) name')
    hdr['INSTRUME'] = ('PSPCC', 'Instrument name')
    hdr['OBS_MODE'] = 'FullFrame'
    hdr['FILTER'] = ('Medium', 'Instrument filter in use')
    today = datetime.date(datetime.now())
    hdr['DATE'] = today.isoformat()
    hdr['RA_OBJ'] = 0.0
    hdr['DEC_OBJ'] = 0.0
    hdr['DATE-OBS'] = today.isoformat()
    hdr['EXPOSURE'] = (tsim_eff, 'Weighted live time of CCDs in the extraction region')
    hdr['CORRFILE'] = 'NONE'
    hdr['CORRSCAL'] = 1.
    hdr['POISSERR'] = (True, 'Poisson errors appropriate')
    hdr['QUALITY'] = 0
    hdr['GROUPING'] = 0
    hdr['RESPFILE'] = rsp_file  # 'none'#(rmf_name, 'redistribution matrix')
    hdr['ANCRFILE'] = 'none'  # (arf_name, 'ancillary response')
    hdr['BACKFILE'] = 'none'  # (bkg_name, 'Background FITS file')
    hdr['CHANTYPE'] = ('PI', 'Type of channel data')
    hdr['DETCHANS'] = len(channel)
    hdr['AREASCAL'] = (1.0, 'Nominal scaling factor for data')
    hdr['BACKSCAL'] = (1.0, 'Scaling factor for background')  # Sum of area in XMM units, 0.05 arcsec
    hdr['CTS'] = np.sum(spectrum).astype(int)
    hdul.append(tbhdu)

    hdul.writeto(name_spec, overwrite=True)
    print('Spectrum written to file', name_spec)

    hdul.close()