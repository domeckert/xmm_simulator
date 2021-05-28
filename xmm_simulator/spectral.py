import numpy as np
from astropy.io import fits
from .utils import region, set_wcs, get_data_file_path
from threeML.utils.OGIP.response import OGIPResponse
from scipy.interpolate import interp1d
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

    pixsize = 30. / box.shape[0]  # arcmin

    # Set region definition
    y, x = np.indices(box[:, :, 0].shape)

    wcs = set_wcs(xmmsim=xmmsim)

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


def save_spectrum(xmmsim, outdir, spectrum, tsim, arf, qpb, backscal, tsim_qpb):
    """

    :param xmmsim:
    :param spectrum:
    :param arf:
    :param qpb:
    :param backscal:
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

    name_spec = outdir + '/'+ pref + '-obj-'+ outdir + '.pi'
    arf_name = outdir + '/'+ pref + '-' + outdir + '.arf'
    bkg_name = outdir + '/'+ pref + '-back-'+ outdir + '.pi'
    rmf_name = outdir + '/'+ pref + '-' + outdir + '.rmf'

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
    hdr['INSTRUME'] = ('EPN', 'Instrument name')
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
    hdr['INSTRUME'] = 'EPN'
    hdr['OBS_MODE'] = 'FullFrame'
    hdr['FILTER'] = 'Medium'
    hdr['DATE'] = today.isoformat()
    hdul.append(tbhdu)

    hdul.writeto(arf_name, overwrite=True)
    print('ARF written to file', arf_name)

    hdul.close()












