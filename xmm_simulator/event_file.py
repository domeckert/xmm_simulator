import numpy as np
from astropy.io import fits
from .utils import get_data_file_path
from .imaging import psf_convole, psf_convole_evt
from .background import tot_area
from scipy.interpolate import interp1d, RectBivariateSpline
from threeML.utils.OGIP.response import OGIPResponse
from threeML import APEC, Powerlaw, PhAbs
import progressbar
from datetime import datetime
import h5py as h5
from tqdm import tqdm
import time
from collections import defaultdict
import gc

lhb_ref = 2.92859e-06 # MACS 0949 sky bkg parameters
ght_ref = 0.220899
ghn_ref = 5.02297e-07
cxb_m13_unres = 1.9411e-07  # unresolved CXB fraction at a limiting flux of 1e-15 in the soft band from Moretti+03
cxb_ref = 7.94099e-07
NH_ref = 0.05

def gen_phot_box(xmmsim, tsim, with_skybkg=True, lhb=None, ght=None, ghn=None, cxb=None, NH=None, abund='angr'):
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
        if xmmsim.pts:
            cxb = cxb_m13_unres

        else:
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

    #pixsize_ori = xmmsim.box_size / xmmsim.box.shape[1] # arcmin

    cx, cy = npix_out / 2., npix_out / 2.

    skybkg_spectrum = None

    if with_skybkg:
        modlhb = APEC()
        modgh = APEC()
        if abund=='aspl':
            modlhb.abundance_table = 'Lodd09'
            modgh.abundance_table = 'Lodd09'

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
        if abund == 'aspl':
            modphabs.abundance_table = 'ASPL'

        #modphabs.init_xsect()

        modphabs.NH = NH

        modsource = xmmsim.pixsize_ori**2 * (modlhb + modphabs * (modgh + modcxb))

        skybkg_spectrum = modsource(xmmsim.box_ene_mean)

    # Get photons per channel
    phot_box = xmmsim.box * tsim * xmmsim.all_arfs

    if xmmsim.pts:
        phot_box = phot_box + xmmsim.box_pts

    if with_skybkg:
        for i in range(len(xmmsim.box_ene_mean)):

            phot_box[:, :, i] = phot_box[:, :, i] + skybkg_spectrum[i] * tsim * xmmsim.all_arfs[:, :, i]

    nene = len(ene) - 1

    xnew = (np.arange(0, npix_out, 1) - cx) * pixsize / xmmsim.pixsize_ori

    ynew = (np.arange(0, npix_out, 1) - cy) * pixsize / xmmsim.pixsize_ori

    phot_box_ima = np.empty((npix_out,npix_out,nene))

    bar = progressbar.ProgressBar()

    for i in bar(range(nene)):
        # Select box data in the chosen energy band
        ima = phot_box[:, :, i]

        # Recast box shape into output image shape
        cx_ori, cy_ori =  xmmsim.box.shape[1]/2. , xmmsim.box.shape[0]/2.

        finterp = RectBivariateSpline(yori - cy_ori, xori - cx_ori, ima.T)

        ima_newpix = finterp(xnew, ynew).T * (pixsize / xmmsim.pixsize_ori)**2 # phot/cm2/s/keV

        phot_box_ima[:,:,i] = psf_convole(ima_newpix, pixsize, xmmsim) * mask

    return phot_box_ima


def sample_rmf(energies, energy_to_channel, rmf_matrix_transposed, emb):
    '''
    Function to convolve an array of energies with the RMF
    '''
    evts_clu_out_rmf = np.empty(len(energies), dtype=emb.dtype)
    for bb, evt in enumerate(energies):
        channel = energy_to_channel(evt)
        prob_dist = rmf_matrix_transposed[channel]  # or rmf.matrix.T ? Confirmed with Eckert, it's .T (P(E) tail at low E)
        # Normalize probabilities
        prob_dist /= np.sum(prob_dist)
        # Sample new energy based on the probability distribution
        evts_clu_out_rmf[bb] = np.random.choice(emb, p=prob_dist)
    return evts_clu_out_rmf



def gen_phot_evtlist(xmmsim, tsim, with_skybkg=True, lhb=None, ght=None, ghn=None, cxb=None, NH=None, abund='angr'):
    """
    Generate a box expectation value in photon/keV , convolved with the PSF and multiplied by the detector mask
    starting from an ideal event list (e.g. from pyxsim)
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
        if xmmsim.pts:
            cxb = cxb_m13_unres

        else:
            cxb = cxb_ref

    if NH is None:
        NH = NH_ref


    if xmmsim.all_arfs is None:
        print('ARF cube not found, please compute ARFs first')

        return

    #Get RMF file
    rmf_file = get_data_file_path('rmfs/%s.rmf' % (xmmsim.instrument))
    rmf = OGIPResponse(rsp_file=rmf_file)
    nchan = len(rmf.monte_carlo_energies) - 1
    mc_ene = (rmf.monte_carlo_energies[:nchan] + rmf.monte_carlo_energies[1:]) / 2.
    bin_width = rmf.monte_carlo_energies[1:] - rmf.monte_carlo_energies[:nchan]
    emb = (rmf.ebounds[1:] + rmf.ebounds[:-1]) / 2.
    ene_to_chan = rmf.energy_to_channel
    rmf_mat_T = rmf.matrix.T

    # Get mask file
    mask_file = get_data_file_path('imgs/%s_mask.fits.gz' % (xmmsim.instrument))
    inmask = fits.open(mask_file)
    mask = inmask[1].data
    pixsize = inmask[1].header['CDELT2'] * 60.  # arcmin
    inmask.close()
    npix_out = mask.shape[0]
    ene = xmmsim.box_ene

    xori = np.arange(0, xmmsim.boxshape1, 1)
    yori = np.arange(0, xmmsim.boxshape0, 1)

    #pixsize_ori = xmmsim.box_size / xmmsim.box.shape[1] # arcmin
    #pixsize_ori = xmmsim.box_size / xmmsim.boxshape1 # arcmin

    cx, cy = npix_out / 2., npix_out / 2.

    skybkg_spectrum = None

    if with_skybkg:
        modlhb = APEC()
        modgh = APEC()
        if abund=='aspl':
            modlhb.abundance_table = 'Lodd09'
            modgh.abundance_table = 'Lodd09'

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
        if abund == 'aspl':
            modphabs.abundance_table = 'ASPL'

        #modphabs.init_xsect()

        modphabs.NH = NH

        modsource = xmmsim.pixsize_ori**2 * (modlhb + modphabs * (modgh + modcxb))

        skybkg_spectrum = modsource(xmmsim.box_ene_mean)

    # Get photons per channel
    ############################## START AGN+BKG ##############################
    #rseppi: Instead, create an empty photbox to store the AGN and the BKG
    #phot_box = xmmsim.box * tsim * xmmsim.all_arfs
    phot_box = np.zeros((xmmsim.boxshape0, xmmsim.boxshape1, len(ene)-1))  #len(ene)-1 ? Yes because ene is the edges

    if xmmsim.pts:
        phot_box = phot_box + xmmsim.box_pts

    if with_skybkg:
        for i in range(len(xmmsim.box_ene_mean)):

            phot_box[:, :, i] = phot_box[:, :, i] + skybkg_spectrum[i] * tsim * xmmsim.all_arfs[:, :, i]

    nene = len(ene) - 1

    xnew = (np.arange(0, npix_out, 1) - cx) * pixsize / xmmsim.pixsize_ori

    ynew = (np.arange(0, npix_out, 1) - cy) * pixsize / xmmsim.pixsize_ori

    phot_box_ima = np.empty((npix_out,npix_out,nene))

    pts_bkg_bool = False
    if np.sum(phot_box)>0:
        pts_bkg_bool = True
        print('Working BKG and AGN: PSF...')
        bar = progressbar.ProgressBar()

        for i in bar(range(nene)):
            # Select box data in the chosen energy band
            ima = phot_box[:, :, i]

            # Recast box shape into output image shape
            #cx_ori, cy_ori =  xmmsim.box.shape[1]/2. , xmmsim.box.shape[0]/2.
            cx_ori, cy_ori =  xmmsim.boxshape1/2. , xmmsim.boxshape0/2.

            finterp = RectBivariateSpline(yori - cy_ori, xori - cx_ori, ima.T)

            ima_newpix = finterp(xnew, ynew).T * (pixsize / xmmsim.pixsize_ori)**2 # phot/cm2/s/keV

            phot_box_ima[:,:,i] = psf_convole(ima_newpix, pixsize, xmmsim) * mask


        #start of gen_evt_list
        ima_tot = np.sum(phot_box_ima, axis=2) * xmmsim.box_ene_width
        print('min ima_tot:', np.min(ima_tot))
        ima_tot[ima_tot<0]=0
        photon_map = np.random.poisson(ima_tot)

        yp, xp = np.indices(photon_map.shape)

        nonz = np.where(photon_map > 0)

        X_evt = np.repeat(xp[nonz], photon_map[nonz])

        Y_evt = np.repeat(yp[nonz], photon_map[nonz])

        chan_evt = np.array([])

        npix_active = len(xp[nonz])

        print('Working on BKG and AGN: RMF...')
        bar = progressbar.ProgressBar()
        chan_evt = []
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

            evts = np.random.choice(emb, p=spec_prob, size=nphot_pix)

            chan_evt.append(evts)
        chan_evt = np.concatenate(chan_evt)
        ############################## END AGN+BKG ##############################
        print('Nevt from BKG and PNT:', len(chan_evt))

    #NOW X_evt, Y_evt, chan_evt contain AGN and BKG events
    #Let's add the cluster events by downgrading pyxsim input events
    #Read evtfile from pyxsim and its properties
    t0 = time.time()
    with h5.File(xmmsim.evtfile_input, 'r') as f_pyxsim:
        eff_area = f_pyxsim['parameters']['area'][()]
        texp_evt = f_pyxsim['parameters']['exp_time'][()]
        x_pix_out = np.floor(f_pyxsim['data']['xsky'][()] * 60. / pixsize + cx).astype(int)
        y_pix_out = np.floor(f_pyxsim['data']['ysky'][()] * 60. / pixsize + cy).astype(int)
        energy = f_pyxsim['data']['eobs'][()]
    texp_ratio = tsim / texp_evt

    xoffset = int(xori[int(len(xori)/2.)])
    yoffset = int(xori[int(len(yori)/2.)])
    print('xoffset:', xoffset)
    print('yoffset:', yoffset)
    print('cx:', cx)
    print('cy:', cy)
    # position in pixels = position in deg / pixsize
    #and also shift 0,0 coordinates in x_pix to the central 256,256 pixel

    #prepare arrays to output pixels and energy of cluster evts
    X_clu, Y_clu, chan_evt_clu = [], [], []
    #Loop on the output pixels
    print('Working on CLU...')
    print('EFFAREA:', eff_area)
    print('Texp:', texp_evt)
    Nstart = len(energy)
    print('Nevt to start:', Nstart)

    print('pixsize',pixsize)
    print('pixsize_ori',xmmsim.pixsize_ori)


    #Add PSF
    print('Adding PSF...')
    x_clu_blurred, y_clu_blurred = psf_convole_evt(x_pix_out, y_pix_out, pixsize, xmmsim)
    sel = (x_clu_blurred>0) & (x_clu_blurred<899) & (y_clu_blurred>0) & (y_clu_blurred<899)
    x_clu_blurred, y_clu_blurred = x_clu_blurred[sel], y_clu_blurred[sel]
    energy = energy[sel]
    ra_pix_blurred = (x_clu_blurred - cx) * pixsize
    dec_pix_blurred = (y_clu_blurred - cy) * pixsize
    x_pix_blurred = np.floor(ra_pix_blurred / xmmsim.pixsize_ori + xoffset).astype(int)
    y_pix_blurred = np.floor(dec_pix_blurred / xmmsim.pixsize_ori + yoffset).astype(int)
    N_blurred = len(x_pix_blurred)
    print('Kept', N_blurred, 'evts, fraction:',N_blurred/Nstart)
    t1 = time.time()
    print('It took', t1-t0, 'sec')


    #Apply camera mask
    print('Applying camera mask...')
    valid_photons = mask[y_clu_blurred, x_clu_blurred] == 1  # Boolean mask
    x_pix_blurred = x_pix_blurred[valid_photons]
    y_pix_blurred = y_pix_blurred[valid_photons]
    x_clu_blurred = x_clu_blurred[valid_photons]
    y_clu_blurred = y_clu_blurred[valid_photons]
    energy = energy[valid_photons]
    N_mask = len(energy)
    print('Kept', N_mask, 'evts, fraction:',N_mask/N_blurred, 'to total:',N_mask/Nstart)
    t2_ = time.time()
    print('Took',t2_-t1,'sec, total:',t2_-t0)

    print('Building photon map...')
    photon_map = defaultdict(list)
    for x_p, y_p, e, x_cl, y_cl in zip(x_pix_blurred, y_pix_blurred, energy, x_clu_blurred, y_clu_blurred):
        photon_map[(x_p, y_p)].append((e, x_cl, y_cl))
    for key in photon_map:
        photon_map[key] = np.array(photon_map[key])
    t2 = time.time()

    del x_pix_blurred
    del y_pix_blurred
    del energy
    del x_clu_blurred
    del y_clu_blurred
    gc.collect()

    print('Took',t2-t2_,'sec, total:',t2-t0)

    for ii,xx in tqdm(enumerate(xori), total=len(xori)): #loop on xori because that's where the arf is computed
        for jj,yy in enumerate(yori):
            #Select the events falling within this pixel
            pix_entries = photon_map.get((xx, yy), [])
            if len(pix_entries) == 0:
                continue
            energy_pix = pix_entries[:, 0]
            x_clu_blurred_pix = pix_entries[:, 1].astype(int)
            y_clu_blurred_pix = pix_entries[:, 2].astype(int)

            #Loop on energy bounds
            for kk,(elow, ehigh) in enumerate(zip(xmmsim.box_ene[:-1], xmmsim.box_ene[1:])):
                #Select events within these bounds
                sel_ene = (energy_pix>=elow) & (energy_pix<ehigh)
                sel_idx = np.where(sel_ene)[0]
                if len(sel_idx) == 0:
                    continue
                evts_this_ene = energy_pix[sel_ene]

                #select fraction of evts based on ARFs ratio and texp ratio
                selfrac = xmmsim.all_arfs[yy,xx,kk]/eff_area * texp_ratio #make sure you select correct arf here
                N_evts_out = np.random.poisson(len(evts_this_ene)*selfrac )
                if N_evts_out>len(evts_this_ene):
                    replace = True
                else:
                    replace = False
                ids = np.random.choice(range(len(evts_this_ene)), size=N_evts_out, replace=replace)
                final_idx = sel_idx[ids]
                evts_clu_out = evts_this_ene[ids]
                if len(evts_clu_out)>0:
                    #Loop on evts to apply rmf
                    evts_clu_out_rmf = sample_rmf(evts_clu_out, ene_to_chan, rmf_mat_T, emb)

                    xx_out = x_clu_blurred_pix[final_idx]
                    yy_out = y_clu_blurred_pix[final_idx]
                    X_clu.append(xx_out)
                    Y_clu.append(yy_out)
                    chan_evt_clu.append(evts_clu_out_rmf)
        if ii%10==0:
            tt = time.time()
            print('Up to xx=',ii,':',tt-t2, 'total:',tt-t0)
            print('Cumulative N CLU evt pix:', len(X_clu))

    del photon_map
    gc.collect()

    # Concatenate
    X_clu = np.concatenate(X_clu)
    Y_clu = np.concatenate(Y_clu)
    chan_evt_clu = np.concatenate(chan_evt_clu)

    print('N CLU evt:', len(X_clu))
    if pts_bkg_bool:
        X_evt = np.append(X_evt, X_clu)
        Y_evt = np.append(Y_evt, Y_clu)
        chan_evt = np.append(chan_evt, chan_evt_clu)
        return X_evt, Y_evt, chan_evt
    else:
        return X_clu, Y_clu, chan_evt_clu

# test by simulating the number of photons first and then drawing the energy
def gen_evt_list(xmmsim, phot_box_ima):
    # Read RMF
    rmf_file = get_data_file_path('rmfs/%s.rmf' % (xmmsim.instrument))

    rmf = OGIPResponse(rsp_file=rmf_file)

    nchan = len(rmf.monte_carlo_energies) - 1

    mc_ene = (rmf.monte_carlo_energies[:nchan] + rmf.monte_carlo_energies[1:]) / 2.

    bin_width = rmf.monte_carlo_energies[1:] - rmf.monte_carlo_energies[:nchan]

    emb = (rmf.ebounds[1:] + rmf.ebounds[:-1]) / 2.

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

        evts = np.random.choice(emb, p=spec_prob, size=nphot_pix)

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

def load_events(xmmsim, infile):
    '''
    Load events from a previous run into the provided XMMSimulator object

    :param xmmsim:
    :param infile:
    :return:
    '''

    fin = fits.open(infile)
    din = fin[1].data

    xmmsim.X_evt = din['X']
    xmmsim.Y_evt = din['Y']
    xmmsim.chan_evt = din['ENERGY']
    xmmsim.time_evt = din['TIME']
    xmmsim.tsim = fin[1].header['EXPOSURE']
    xmmsim.events = True
