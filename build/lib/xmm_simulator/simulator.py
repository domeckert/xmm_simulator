import numpy as np
from astropy.io import fits
from scipy.interpolate import interp1d, interp2d
import os
from astropy.cosmology import FlatLambdaCDM
import progressbar
import pkg_resources

cosmo_elena = FlatLambdaCDM(Om0=0.307114989, H0=67.77)

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

    fvig_ebound = interp1d(ene_vig, vig_theta, kind='cubic')

    vig_ebound = fvig_ebound(ebound)

    fcorr_interp = interp1d(ene_corr, corr_fact, kind='cubic')

    corr_ebound = fcorr_interp(ebound)

    fqeff = interp1d(xmmsim.ene_qeff.flatten(), xmmsim.qeff.flatten(), kind='cubic', fill_value="extrapolate")

    qeff_ebound = fqeff(ebound)

    filter_interp = interp1d(xmmsim.ene_filter.flatten(), xmmsim.filter.flatten(), kind='cubic', fill_value="extrapolate")

    filter_ebound = filter_interp(ebound)

    arf = area_ebound * vig_ebound * corr_ebound * filter_ebound * qeff_ebound

    return arf


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


class XMMSimulator(object):
    """

    """
    def __init__(self, boxfile, ccfpath, instrument):
        """
        Constructor of class XMMSimulator

        :param boxfile: Input photon box file
        :param ccfpath: Path to calibration file directory
        :param instrument: Instrument to be simulated (PN, MOS1, or MOS2)
        """

        fb = fits.open(boxfile)

        self.box = fb[0].data

        fb.close()

        self.box_ene = np.arange(0.1, 10.02, 0.02)

        self.box_ene_mean = np.arange(0.11, 10., 0.02)

        self.box_ene_width = 0.02

        self.ccfpath = ccfpath

        try:
            instrument == 'PN' or instrument=='MOS1' or instrument=='MOS2'

        except TypeError:
            print('ERROR: instrument should be one of PN, MOS1, or MOS2')

        self.instrument = instrument

        get_ccf_file_names(self)

        ccf_arf_pn = fits.open(ccfpath + self.area_file)
        self.onaxis = ccf_arf_pn[1].data
        self.vignetting = ccf_arf_pn[2].data
        self.corrarea = ccf_arf_pn[3].data

        self.dtheta = ccf_arf_pn[2].header['D_THETA'] * 60.

        self.nrad = len(self.vignetting[0][1])
        self.rads_vignetting = np.arange(0., 16., self.dtheta)

        fqeff = fits.open(ccfpath + self.qe_file)

        self.ene_qeff = fqeff['EBINS_FRACTION'].data['ENERGY'] / 1e3

        self.qeff = fqeff['QE_TOTAL'].data['QE_TOTAL']

        fqeff.close()

        filter_transf = fits.open(ccfpath + self.filter_file)

        self.ene_filter = filter_transf['EBINS'].data['ENERGY'] / 1e3

        self.filter = filter_transf['FILTER-MEDIUM'].data['TRANSMISSION']

        filter_transf.close()


    def ARF_Box(self):
        """
        Compute the ARFs for each point on the box

        """

        pixsize = 30. / self.box.shape[0]  # box size is 30 arcmin

        y, x = np.indices(self.box[:, :, 0].shape)

        cx, cy = self.box.shape[0] / 2., self.box.shape[1] / 2.

        self.cx = cx
        self.cy = cy

        thetas = np.hypot(x - cx, y - cy) * pixsize  # arcmin

        nene_ori = self.box.shape[2]

        ene_lo, ene_hi = self.box_ene[:nene_ori], self.box_ene[1:]

        all_arfs = np.empty((self.box.shape[0], self.box.shape[1], nene_ori))

        for i in progressbar.progressbar(range(self.box.shape[0])):

            for j in range(self.box.shape[1]):

                if thetas[j, i] <= np.max(self.rads_vignetting):

                    all_arfs[j, i, :] = calc_arf(thetas[j, i], ene_lo, ene_hi, self)


        self.all_arfs = all_arfs

    def ExtractImage(self, outfile, elow=0.5, ehigh=2.0,):
        """
        Extract an image in the energy band (elow, ehigh). The image is Poissonized, convolved with PSF, and background is added.

        :param outfile: Name of output FTTS file
        :type outfile: str
        :param elow: Lower energy boundary of the image
        :type elow: float
        :param ehigh: Upper energy boundary of the image
        :type ehigh: float
        """



