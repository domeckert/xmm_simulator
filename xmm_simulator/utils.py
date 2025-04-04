import pkg_resources
import os
from scipy.interpolate import interp1d, interp2d
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from threeML.utils.OGIP.response import OGIPResponse

def calc_arf(theta, ebound_lo, ebound_hi, xmmsim):
    """
    Function to compute the ARF at a given off-axis angle theta

    :param theta: Off-axis angle in arcmin
    :type theta: float
    :param ebound_lo: Array containing the lower boundaries of the energy bins
    :type ebound_lo: numpy.ndarray
    :param ebound_hi: Array containing the upper boundaries of the energy bins
    :type ebound_hi: numpy.ndarray
    :param onaxis: On-axis response
    :param vignetting: Vignetting curve and energy dependence
    :param dtheta: Size of vignetting bins in arcmin
    :param corrarea: Effective area correction factor
    :param ene_qeff: Quantum efficiency energy definition
    :param qeff: Quantum efficiency curve
    :param ene_filter: Filter energy definition
    :param filter_transf: Filter transmission curve
    :return: ARF
    :rtype: numpy.ndarray
    """
    dtheta = xmmsim.dtheta

    rads = np.arange(0., 16., dtheta)

    ene_vig = xmmsim.vignetting['ENERGY'] / 1e3
    vig_fact = xmmsim.vignetting['VIGNETTING_FACTOR']

    ene_onaxis = xmmsim.onaxis['ENERGY'] / 1e3
    area_onaxis = xmmsim.onaxis['AREA']

    ene_corr = xmmsim.corrarea['ENERGY'] / 1e3
    corr_fact = xmmsim.corrarea['FACTOR']

    fvig_interp = interp2d(rads, ene_vig, vig_fact, kind='cubic')

    vig_theta = fvig_interp(theta, ene_vig).flatten()

    ebound = (ebound_lo + ebound_hi) / 2.

    farea_interp = interp1d(ene_onaxis, area_onaxis, kind='linear')

    area_ebound = farea_interp(ebound)

    if 'MOS' in xmmsim.instrument:

        fareacorr = interp1d(xmmsim.ene_areacorr, xmmsim.areacorr, kind='linear', fill_value='extrapolate')

        areacorr_ebound = fareacorr(ebound)

        area_ebound = area_ebound * areacorr_ebound

    fvig_ebound = interp1d(ene_vig, vig_theta, kind='cubic')

    vig_ebound = fvig_ebound(ebound)

    fcorr_interp = interp1d(ene_corr, corr_fact, kind='cubic')

    corr_ebound = fcorr_interp(ebound)

    fqeff = interp1d(xmmsim.ene_qeff.flatten(), xmmsim.qeff.flatten(), kind='cubic', fill_value="extrapolate")

    qeff_ebound = fqeff(ebound)

    filter_interp = interp1d(xmmsim.ene_filter.flatten(), xmmsim.filter.flatten(), kind='cubic', fill_value="extrapolate")

    filter_ebound = filter_interp(ebound)

    arf = area_ebound * vig_ebound * corr_ebound * filter_ebound * qeff_ebound

    if np.any(arf<0.):

        print('WARNING: some negative ARF value detected, set to zero')

        negarf = np.where(arf<0.)

        arf[negarf] = 0.

    if np.any(np.isnan(arf)):

        print('WARNING: some NaN ARF value detected, set to zero')

        nanarf = np.where(np.isnan(arf))

        arf[nanarf] = 0.

    return arf


def get_data_file_path(data_file):
    """
    Returns the absolute path to the required data files.
    :param data_file: relative path to the data file, relative to the astromodels/data path.
    So to get the path to data/dark_matter/gammamc_dif.dat you need to use data_file="dark_matter/gammamc_dif.dat"
    :return: absolute path of the data file
    """

    try:

        file_path = pkg_resources.resource_filename("xmm_simulator", '%s' % data_file)

    except KeyError:

        raise IOError("Could not read or find data file %s." % (data_file))

    else:

        return os.path.abspath(file_path)



def get_ccf_file_names(xmmsim):
    """
    Function that searches through the CCF directory and selects the latest calibration files

    :param xmmsim: XMM Simulator
    :type xmmsim: class:`xmm_simulator.XMMSimulator`
    """

    ccfpath = xmmsim.ccfpath

    instrument = xmmsim.instrument

    all_ccf = os.listdir(ccfpath)

    nqe, narea, nfilter, nfwc, npsf = 0, 0, 0, 0, 0

    for tf in all_ccf:

        if 'E' + instrument + '_QUANTUMEF' in tf:

            tn = int(tf.split('_')[2].split('.')[0])

            if tn > nqe:
                nqe = tn

                xmmsim.qe_file = tf

        if 'E' + instrument + '_FILTERTRANSX' in tf:

            tn = int(tf.split('_')[2].split('.')[0])

            if tn > nfilter:
                nfilter = tn

                xmmsim.filter_file = tf

        if 'E' + instrument + '_FWC' in tf:

            tn = int(tf.split('_')[2].split('.')[0])

            if tn > nfwc:
                nfwc = tn

                xmmsim.fwc_file = tf

        if instrument == 'PN':

            if 'XRT3_XAREAEF' in tf:

                tn = int(tf.split('_')[2].split('.')[0])

                if tn > narea:
                    narea = tn

                    xmmsim.area_file = tf

            if 'XRT3_XPSF' in tf:

                tn = int(tf.split('_')[2].split('.')[0])

                if tn > npsf:
                    npsf = tn

                    xmmsim.psf_file = tf

        if instrument == 'MOS1':

            if 'XRT1_XAREAEF' in tf:

                tn = int(tf.split('_')[2].split('.')[0])

                if tn > narea:
                    narea = tn

                    xmmsim.area_file = tf

            if 'XRT1_XPSF' in tf:

                tn = int(tf.split('_')[2].split('.')[0])

                if tn > npsf:
                    npsf = tn

                    xmmsim.psf_file = tf

        if instrument == 'MOS2':

            if 'XRT1_XAREAEF' in tf:

                tn = int(tf.split('_')[2].split('.')[0])

                if tn > narea:
                    narea = tn

                    xmmsim.area_file = tf

            if 'XRT2_XPSF' in tf:

                tn = int(tf.split('_')[2].split('.')[0])

                if tn > npsf:
                    npsf = tn

                    xmmsim.psf_file = tf

def set_wcs(xmmsim, type='mask'):
    """
    Get the fake (but consistent) WCS information for the box image

    :param xmmsim: XMMSimulator
    :return: astropy WCS class
    """
    # Get mask file
    mask_file = get_data_file_path('imgs/%s_mask.fits.gz' % (xmmsim.instrument))

    inmask = fits.open(mask_file)

    head_mask = inmask[1].header

    mask = inmask[1].data

    inmask.close()

    npix_mask = mask.shape[0]

    #npix_box = ima.shape[0]
    npix_box = xmmsim.boxshape0

    if type == 'box':
        pixsize_ori = xmmsim.box_size / 60. / xmmsim.boxshape1  # degree

        if xmmsim.box is not None:
            ima = xmmsim.box[:, :, 0]
            hdu = fits.PrimaryHDU(ima)
            header = hdu.header
        else:
            header = fits.Header()

        header['CDELT1'] = - pixsize_ori
        header['CDELT2'] = pixsize_ori
        header['CRPIX1'] = head_mask['CRPIX1'] * npix_box / npix_mask
        header['CRPIX2'] = head_mask['CRPIX2'] * npix_box / npix_mask
        header['CTYPE1'] = head_mask['CTYPE1']
        header['CTYPE2'] = head_mask['CTYPE2']
        header['CRVAL1'] = 0.
        header['CRVAL2'] = 0.
        wcs = WCS(header)

    else:

        head_mask['CRVAL1'] = 0.
        head_mask['CRVAL2'] = 0.
        wcs = WCS(head_mask)

    return wcs


def region(regfile, thetas, wcs_inp, pixsize):
    """
    Mask regions selected in regfile

    :param regfile: Region file in DS9 format
    :param thetas:  Array containing the radii to the center in arcmin
    :param wcs_inp: WCS coordinate transformation class
    :param pixsize: Pixel size
    :return: Modified radii with masked regions = -1
    """
    freg = open(regfile)
    lreg = freg.readlines()
    freg.close()
    nsrc = 0
    nreg = len(lreg)

    masked_thetas = np.copy(thetas)

    y, x = np.indices(thetas.shape)

    regtype = None

    for i in range(nreg):
        if 'fk5' in lreg[i]:
            regtype = 'fk5'
        elif 'image' in lreg[i]:
            regtype = 'image'

    if regtype is None:
        print('Error: invalid format')
        return
    for i in range(nreg):
        if 'circle' in lreg[i]:
            vals = lreg[i].split('(')[1].split(')')[0]
            if regtype == 'fk5':
                xsrc = float(vals.split(',')[0])
                ysrc = float(vals.split(',')[1])
                rad = vals.split(',')[2]
                if '"' in rad:
                    rad = float(rad.split('"')[0]) / pixsize / 60.
                elif '\'' in rad:
                    rad = float(rad.split('\'')[0]) / pixsize
                else:
                    rad = float(rad) / pixsize * 60.
                wc = np.array([[xsrc, ysrc]])
                pixcrd = wcs_inp.wcs_world2pix(wc, 1)
                xsrc = pixcrd[0][0] - 1.
                ysrc = pixcrd[0][1] - 1.
            else:
                xsrc = float(vals.split(',')[0])
                ysrc = float(vals.split(',')[1])
                rad = float(vals.split(',')[2])

            # Define box around source to spped up calculation
            boxsize = np.round(rad + 0.5).astype(int)
            intcx = np.round(xsrc).astype(int)
            intcy = np.round(ysrc).astype(int)
            xmin = np.max([intcx - boxsize, 0])
            xmax = np.min([intcx + boxsize + 1, masked_thetas.shape[1]])
            ymin = np.max([intcy - boxsize, 0])
            ymax = np.min([intcy + boxsize + 1, masked_thetas.shape[0]])
            rbox = np.hypot(x[ymin:ymax, xmin:xmax] - xsrc, y[ymin:ymax, xmin:xmax] - ysrc)
            # Mask source
            src = np.where(rbox < rad)
            masked_thetas[ymin:ymax, xmin:xmax][src] = -1.0
            nsrc = nsrc + 1
        elif 'ellipse' in lreg[i]:
            vals = lreg[i].split('(')[1].split(')')[0]
            if regtype == 'fk5':
                xsrc = float(vals.split(',')[0])
                ysrc = float(vals.split(',')[1])
                rad1 = vals.split(',')[2]
                rad2 = vals.split(',')[3]
                angle = float(vals.split(',')[4])
                if '"' in rad1:
                    rad1 = float(rad1.split('"')[0]) / pixsize / 60.
                    rad2 = float(rad2.split('"')[0]) / pixsize / 60.
                elif '\'' in rad1:
                    rad1 = float(rad1.split('\'')[0]) / pixsize
                    rad2 = float(rad2.split('\'')[0]) / pixsize
                else:
                    rad1 = float(rad1) / pixsize * 60.
                    rad2 = float(rad2) / pixsize * 60.
                wc = np.array([[xsrc, ysrc]])
                pixcrd = wcs_inp.wcs_world2pix(wc, 1)
                xsrc = pixcrd[0][0] - 1.
                ysrc = pixcrd[0][1] - 1.
            else:
                xsrc = float(vals.split(',')[0])
                ysrc = float(vals.split(',')[1])
                rad1 = float(vals.split(',')[2])
                rad2 = float(vals.split(',')[3])
                angle = float(vals.split(',')[2])
            ellang = angle * np.pi / 180. + np.pi / 2.
            aoverb = rad1 / rad2
            # Define box around source to spped up calculation
            boxsize = np.round(np.max([rad1, rad2]) + 0.5).astype(int)
            intcx = np.round(xsrc).astype(int)
            intcy = np.round(ysrc).astype(int)
            xmin = np.max([intcx - boxsize, 0])
            xmax = np.min([intcx + boxsize + 1, masked_thetas.shape[1]])
            ymin = np.max([intcy - boxsize, 0])
            ymax = np.min([intcy + boxsize + 1, masked_thetas.shape[0]])
            xtil = np.cos(ellang) * (x[ymin:ymax, xmin:xmax] - xsrc) + np.sin(ellang) * (y[ymin:ymax, xmin:xmax] - ysrc)
            ytil = -np.sin(ellang) * (x[ymin:ymax, xmin:xmax] - xsrc) + np.cos(ellang) * (
                        y[ymin:ymax, xmin:xmax] - ysrc)
            rbox = aoverb * np.hypot(xtil, ytil / aoverb)
            # Mask source
            src = np.where(rbox < rad1)
            masked_thetas[ymin:ymax, xmin:xmax][src] = -1.0
            nsrc = nsrc + 1

    print('Excluded %d sources' % (nsrc))

    return masked_thetas


def region_evt(xmmsim, regfile, thetas, wcs_inp, pixsize):
    """
    Mask regions selected in regfile

    :param regfile: Region file in DS9 format
    :param thetas:  Array containing the radii to the center in arcmin
    :param wcs_inp: WCS coordinate transformation class
    :param pixsize: Pixel size
    :return: Modified radii with masked regions = -1
    """
    freg = open(regfile)
    lreg = freg.readlines()
    freg.close()
    nsrc = 0
    nreg = len(lreg)

    masked_thetas = np.copy(thetas)

    regtype = None

    for i in range(nreg):
        if 'fk5' in lreg[i]:
            regtype = 'fk5'
        elif 'image' in lreg[i]:
            regtype = 'image'

    if regtype is None:
        print('Error: invalid format')
        return
    for i in range(nreg):
        if 'circle' in lreg[i]:
            vals = lreg[i].split('(')[1].split(')')[0]
            if regtype == 'fk5':
                xsrc = float(vals.split(',')[0])
                ysrc = float(vals.split(',')[1])
                rad = vals.split(',')[2]
                if '"' in rad:
                    rad = float(rad.split('"')[0]) / pixsize / 60.
                elif '\'' in rad:
                    rad = float(rad.split('\'')[0]) / pixsize
                else:
                    rad = float(rad) / pixsize * 60.
                wc = np.array([[xsrc, ysrc]])
                pixcrd = wcs_inp.wcs_world2pix(wc, 1)
                xsrc = pixcrd[0][0] - 1.
                ysrc = pixcrd[0][1] - 1.
            else:
                xsrc = float(vals.split(',')[0])
                ysrc = float(vals.split(',')[1])
                rad = float(vals.split(',')[2])

            # mask events that are inside the area
            rads = np.hypot(xsrc - xmmsim.X_evt, ysrc - xmmsim.Y_evt)

            # Mask source
            src = np.where(rads < rad)
            masked_thetas[src] = -1.0

            nsrc = nsrc + 1

        elif 'ellipse' in lreg[i]:
            vals = lreg[i].split('(')[1].split(')')[0]
            if regtype == 'fk5':
                xsrc = float(vals.split(',')[0])
                ysrc = float(vals.split(',')[1])
                rad1 = vals.split(',')[2]
                rad2 = vals.split(',')[3]
                angle = float(vals.split(',')[4])
                if '"' in rad1:
                    rad1 = float(rad1.split('"')[0]) / pixsize / 60.
                    rad2 = float(rad2.split('"')[0]) / pixsize / 60.
                elif '\'' in rad1:
                    rad1 = float(rad1.split('\'')[0]) / pixsize
                    rad2 = float(rad2.split('\'')[0]) / pixsize
                else:
                    rad1 = float(rad1) / pixsize * 60.
                    rad2 = float(rad2) / pixsize * 60.
                wc = np.array([[xsrc, ysrc]])
                pixcrd = wcs_inp.wcs_world2pix(wc, 1)
                xsrc = pixcrd[0][0] - 1.
                ysrc = pixcrd[0][1] - 1.
            else:
                xsrc = float(vals.split(',')[0])
                ysrc = float(vals.split(',')[1])
                rad1 = float(vals.split(',')[2])
                rad2 = float(vals.split(',')[3])
                angle = float(vals.split(',')[2])
            ellang = angle * np.pi / 180. + np.pi / 2.
            aoverb = rad1 / rad2
            xtil = np.cos(ellang) * (xmmsim.X_evt - xsrc) + np.sin(ellang) * (xmmsim.Y_evt - ysrc)
            ytil = -np.sin(ellang) * (xmmsim.X_evt - xsrc) + np.cos(ellang) * (xmmsim.Y_evt - ysrc)
            rads = aoverb * np.hypot(xtil, ytil / aoverb)

            # Mask source
            src = np.where(rads < rad)
            masked_thetas[src] = -1.0
            nsrc = nsrc + 1

    print('Excluded %d sources' % (nsrc))

    return masked_thetas


def arf_region(xmmsim, cra, cdec, rin, rout, regfile=None):
    '''
    Computing the emission-weighted ARF of a given region (annulus or circle) defined by the input parameters

    :param xmmsim:
    :param cra:
    :param cdec:
    :param rin:
    :param rout:
    :param regfile:
    :return:
    '''

    if xmmsim.all_arfs is None:
        print('ARF cube not found, please compute ARFs first')

        return

    box = xmmsim.box

    pixsize = xmmsim.box_size / box.shape[0]  # arcmin

    # Get mask file
    mask_file = get_data_file_path('imgs/%s_mask.fits.gz' % (xmmsim.instrument))

    inmask = fits.open(mask_file)

    mask = inmask[1].data

    pixsize_mask = inmask[1].header['CDELT2'] * 60.  # arcmin

    inmask.close()

    # Set region definition
    y, x = np.indices(box[:, :, 0].shape)

    wcs = set_wcs(xmmsim=xmmsim, type='box')

    wc = np.array([[cra, cdec]])

    pixcrd = wcs.wcs_world2pix(wc, 1)

    xsrc = pixcrd[0][0] - 1.

    ysrc = pixcrd[0][1] - 1.

    thetas = np.hypot(x - xsrc, y - ysrc) * pixsize  # arcmin

    # Recast mask shape into box shape
    x0 = (pixsize_mask * mask.shape[1] - xmmsim.box_size) / 2. / pixsize_mask

    y0 = (pixsize_mask * mask.shape[0] - xmmsim.box_size) / 2. / pixsize_mask

    x_box = x0 + np.arange(0, xmmsim.box.shape[1], 1) * pixsize / pixsize_mask

    y_box = y0 + np.arange(0, xmmsim.box.shape[0], 1) * pixsize / pixsize_mask

    y_near = np.repeat(np.floor(y_box + 0.5).astype(int), xmmsim.box.shape[1]).reshape(xmmsim.box.shape[0],
                                                                                       xmmsim.box.shape[1])

    x_near = np.tile(np.floor(x_box + 0.5).astype(int), xmmsim.box.shape[0]).reshape(xmmsim.box.shape[0],
                                                                                     xmmsim.box.shape[1])

    ind_box = (y_near, x_near)

    mask_box = mask[ind_box]

    if regfile is not None:

        thetas = region(regfile=regfile,
                        thetas=thetas,
                        wcs_inp=wcs,
                        pixsize=pixsize)

    test_annulus = np.where(np.logical_and(np.logical_and(thetas >= rin, thetas < rout), mask_box>0.))

    arfs_sel = xmmsim.all_arfs[test_annulus]

    box_sel = xmmsim.box[test_annulus]

    # Read RMF
    rmf_file = get_data_file_path('rmfs/%s.rmf' % (xmmsim.instrument))

    rmf = OGIPResponse(rsp_file=rmf_file)

    nchan = len(rmf.monte_carlo_energies) - 1

    mc_ene = (rmf.monte_carlo_energies[:nchan] + rmf.monte_carlo_energies[1:]) / 2.

    # Compute mean ARF and interpolate
    arf_mean = np.average(arfs_sel, axis=0, weights=box_sel)

    finterp_arf = interp1d(xmmsim.box_ene_mean, arf_mean, fill_value='extrapolate')

    arf_mean_interp = finterp_arf(mc_ene)

    neg_arf = np.where(arf_mean_interp < 0.)

    arf_mean_interp[neg_arf] = 0.

    return arf_mean_interp
