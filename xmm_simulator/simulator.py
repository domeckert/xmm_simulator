import numpy as np
from astropy.io import fits
import os
from astropy.cosmology import FlatLambdaCDM
import progressbar
from .background import gen_qpb_image, gen_skybkg_image, gen_skybkg_spectrum, gen_qpb_spectrum
from .utils import get_ccf_file_names, calc_arf, get_data_file_path
from .imaging import gen_image_box, save_maps
from .spectral import gen_spec_box, save_spectrum

class XMMSimulator(object):
    """

    """
    def __init__(self, boxfile, ccfpath, instrument):
        """
        Constructor of class XMMSimulator

        :param boxfile: Input photon box file
        :type boxfile: str
        :param ccfpath: Path to calibration file directory
        :type ccfpath: str
        :param instrument: Instrument to be simulated (PN, MOS1, or MOS2)
        :type instrument: str
        """

        try:
            if instrument not in ['PN', 'MOS1', 'MOS2']:
                raise ValueError

        except ValueError:
            print('ERROR: instrument should be one of PN, MOS1, or MOS2')
            return

        fb = fits.open(boxfile)

        self.box = fb[0].data

        fb.close()

        self.box_ene = np.arange(0.1, 10.02, 0.02)

        self.box_ene_mean = np.arange(0.11, 10., 0.02)

        self.box_ene_width = 0.02

        self.ccfpath = ccfpath

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

        areacorr_file = get_data_file_path('rmfs/mos_areacorr.fits')

        fareacorr = fits.open(areacorr_file)

        areacorr_data = fareacorr['AREACORR'].data

        self.ene_areacorr = areacorr_data['ENERGY']

        self.areacorr = areacorr_data['AREACORR']

        fareacorr.close()

        self.cx = None
        self.cy = None
        self.all_arfs = None


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

    def ExtractImage(self, tsim, outname, elow=0.5, ehigh=2.0, nbin=10, withskybkg=True, withqpb=True, lhb=None, ght=None, ghn=None, cxb=None, NH=None):
        """
        Extract an image in the energy band (elow, ehigh). The image is Poissonized, convolved with PSF, and background is added.

        :param tsim: Exposure time of simulation
        :type tsim: float
        :param outname: Name used for output FTTS files
        :type outname: str
        :param elow: Lower energy boundary of the image (defaults to 0.5)
        :type elow: float
        :param ehigh: Upper energy boundary of the image (defaults to 2.0)
        :type ehigh: float
        :param nbin: Number of energy bins into which the calculation will be split for exposure map/vignetting calculation (defaults to 10)
        :type nbin: int
        :param withskybkg: Switch to simulate or not the sky background (defaults to True)
        :type withskybkg: bool
        :param withqpb: Switch to simulate or not the quiescent particle background (defaults to True)
        :type withqpb: bool
        :param lhb: Local Hot Bubble normalization per square arcmin
        :type lhb: float
        :param ght: Galactic Halo temperature in keV
        :type ght: float
        :param ghn: Galactic Halo normalization per square arcmin
        :type ghn: float
        :param cxb: Cosmic X-ray background normalization per square arcmin
        :type cxb: float
        :param NH: Absorption column density
        :type NH: float
        """

        print('# Generating sky background and exposure maps...')
        skybkg_map, expmap = gen_skybkg_image(self,
                                              tsim=tsim,
                                              elow=elow,
                                              ehigh=ehigh,
                                              nbin=nbin,
                                              lhb=lhb,
                                              ght=ght,
                                              ghn=ghn,
                                              cxb=cxb,
                                              NH=NH)

        if not withskybkg:
            skybkg_map = skybkg_map * 0.

        if withqpb:
            print('# Generating QPB map...')
            qpb_map = gen_qpb_image(self,
                                    tsim=tsim,
                                    elow=elow,
                                    ehigh=ehigh)

        else:
            qpb_map = skybkg_map * 0.

        print('# Generating box image...')
        box_map = gen_image_box(self,
                                tsim=tsim,
                                elow=elow,
                                ehigh=ehigh,
                                nbin=nbin)

        print("# Simulating data...")
        poisson_map = np.random.poisson(box_map + qpb_map + skybkg_map)

        print('# Saving data into output files...')
        save_maps(self,
                  outname=outname,
                  countmap=poisson_map,
                  expmap=expmap,
                  bkgmap=qpb_map)

    def ExtractSpectrum(self, tsim, outdir, cra, cdec, rin, rout, regfile=None, withskybkg=True, withqpb=True, lhb=None, ght=None, ghn=None, cxb=None, NH=None):
        """
        Extract the spectrum, ARF and background file for an annulus between rin and rout. Regions can be masked by providing a DS9 region file.

        :param tsim: Exposure time of simulation
        :type tsim: float
        :param outdir: Name of output directory
        :type outdir: str
        :param cra: Right ascension of the center of the annulus
        :type cra: float
        :param cdec: Declination of the center of the annulus
        :type cdec: float
        :param rin: Inner radius of the region in arcmin
        :type rin: float
        :param rout: Outer radius of the region in arcmin
        :type rout: float
        :param regfile: DS9 region file containing the definition of regions to be excluded
        :type regfile: str
        :param withskybkg: Switch to simulate or not the sky background (defaults to True)
        :type withskybkg: bool
        :param withqpb: Switch to simulate or not the quiescent particle background (defaults to True)
        :type withqpb: bool
        :param lhb: Local Hot Bubble normalization per square arcmin
        :type lhb: float
        :param ght: Galactic Halo temperature in keV
        :type ght: float
        :param ghn: Galactic Halo normalization per square arcmin
        :type ghn: float
        :param cxb: Cosmic X-ray background normalization per square arcmin
        :type cxb: float
        :param NH: Absorption column density
        :type NH: float
        """

        print('# Extracting spectrum...')
        box_spec, arf, backscal = gen_spec_box(self,
                                               tsim=tsim,
                                               cra=cra,
                                               cdec=cdec,
                                               rin=rin,
                                               rout=rout,
                                               regfile=regfile)

        area_spec = backscal / 60.**2 * (0.05**2) # arcmin^2

        if withqpb:
            print('# Extracting FWC spectrum...')
            qpb_spec = gen_qpb_spectrum(self,
                                        tsim=tsim,
                                        area_spec=area_spec)
        else:
            qpb_spec = box_spec * 0.

        if withskybkg:
            print('# Generating sky background spectrum...')
            skybkg_spec = gen_skybkg_spectrum(self,
                                              tsim=tsim,
                                              area_spec=area_spec,
                                              arf=arf,
                                              lhb=lhb,
                                              ght=ght,
                                              ghn=ghn,
                                              cxb=cxb,
                                              NH=NH)
        else:
            skybkg_spec = box_spec * 0.

        spectrum = np.random.poisson(box_spec + skybkg_spec) + qpb_spec # Total spectrum

        print('# Now saving spectra and region files...')
        save_spectrum(self,
                      outdir=outdir,
                      spectrum=spectrum,
                      tsim=tsim,
                      arf=arf,
                      qpb=qpb_spec,
                      backscal=backscal)
