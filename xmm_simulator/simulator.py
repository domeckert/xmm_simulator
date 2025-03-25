import numpy as np
from astropy.io import fits
import os
from astropy.cosmology import FlatLambdaCDM
from .background import gen_qpb_image, gen_skybkg_image, gen_skybkg_spectrum, gen_qpb_spectrum, tot_area, read_qpb_spectrum
from .utils import get_ccf_file_names, calc_arf, get_data_file_path
from .imaging import gen_image_box, save_maps, exposure_map
from .spectral import gen_spec_box, save_spectrum, gen_spec_evt, gen_spec_evt_pix
from .event_file import gen_phot_box, gen_evt_list, gen_qpb_evt, merge_evt, save_evt_file, gen_image_evt, load_events, gen_phot_evtlist
from .point_sources import gen_sources, pts_box
from scipy.interpolate import interp1d

class XMMSimulator(object):
    """

    """
    def __init__(self, ccfpath, instrument, tsim, boxfile=None,
                 evtfile_input=None, box_size=0.5, box_ene=None, abund='angr'):
        """
        Constructor of class XMMSimulator

        :param boxfile: Input photon box file
        :type boxfile: str
        :param ccfpath: Path to calibration file directory
        :type ccfpath: str
        :param instrument: Instrument to be simulated (PN, MOS1, or MOS2)
        :type instrument: str
        :param tsim: Simulation exposure time
        :type tsim: float
        :param evtfile_input: Input event file (in the format of output event file from pyxsim)
        :type evtfile_input: str
        :param box_size: Size of the provided simulation box in degrees (defaults to 0.5)
        :type box_size: float
        :param box_ene: Numpy array containing the energy definition of the box. If None, defaults to a linear grid between 0.1 and 10 keV with a step of 0.02
        :type box_ene: numpy.ndarray
        :param abund: Solar abundance table. Can be set to 'angr' (Anders & Grevesse 1988) or 'aspl' (Asplund et al. 2009). Defaults to 'angr'
        :type abund: str
        """

        try:
            if instrument not in ['PN', 'MOS1', 'MOS2']:
                raise ValueError

        except ValueError:
            print('ERROR: instrument should be one of PN, MOS1, or MOS2')
            return

        try:
            if boxfile is None and evtfile_input is None:
                raise ValueError
        except ValueError:
            print('ERROR: you need to provide at least a path to a boxfile or a path to an input eventfile')

        try:
            if boxfile is not None and evtfile_input is not None:
                raise ValueError
        except ValueError:
            print('ERROR: you provided a path to a boxfile and a path to an input eventfile. Choose one of the two')


        self.box_size = box_size * 60.  # arcmin
        if boxfile:
            fb = fits.open(boxfile)

            self.box = fb[0].data

            self.pixsize = self.box_size / self.box.shape[0]

            self.boxshape0 = self.box.shape[0]
            self.boxshape1 = self.box.shape[1]
            self.boxshape2 = self.box.shape[2]

            fb.close()
            self.evtfile_input = None
        else:
            self.box = None
            self.evtfile_input = evtfile_input
            self.boxshape0 = 100#512
            self.boxshape1 = 100#512
            self.boxshape2 = 99#495
            self.pixsize = self.box_size / self.boxshape0


        if box_ene is None:
            #self.box_ene = np.arange(0.1, 10.01, 0.02)
            #self.box_ene_mean = np.arange(0.11, 10., 0.02)
            #self.box_ene_width = 0.02
            self.box_ene = np.linspace(0.1, 10.01, self.boxshape2+1)
            self.box_ene_mean = np.linspace(0.11, 10., self.boxshape2)
            self.box_ene_width = np.diff(self.box_ene_mean)[0]
        else:
            self.box_ene = box_ene

            self.box_ene_mean = (box_ene[1:] + box_ene[:-1]) / 2.

            self.box_ene_width = box_ene[1:] - box_ene[:-1]

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
        ccf_arf_pn.close()

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

        rmf_file = get_data_file_path('rmfs/%s.rmf' % (instrument))

        frmf = fits.open(rmf_file)
        ebounds = frmf['EBOUNDS'].data
        frmf.close()
        self.ebounds = ebounds

        self.tsim = tsim

        self.cx = None
        self.cy = None
        self.all_arfs = None
        self.fwc_spec = None
        self.events = False
        self.pts = False
        if abund!='angr' and abund!='aspl':
            print('Unknown abundance table %s, defaulting to angr')
            self.abund = 'angr'
        else:
            self.abund = abund

    def ARF_Box(self):
        """
        Compute the ARFs for each point on the box

        """
        #redefined by rseppi as self.pixsize
        #pixsize = self.box_size / self.box.shape[0]  # by default box size is 30 arcmin

        #y, x = np.indices(self.box[:, :, 0].shape)
        y, x = np.indices((self.boxshape0, self.boxshape1))

        #cx, cy = self.box.shape[0] / 2., self.box.shape[1] / 2.
        cx, cy = self.boxshape0 / 2., self.boxshape1 / 2.

        self.cx = cx
        self.cy = cy

        nthetas = 200

        thetas = np.linspace(0., 22., nthetas)

        theta_image = np.hypot(x - cx, y - cy) * self.pixsize  # arcmin

        #nene_ori = self.box.shape[2]
        nene_ori = self.boxshape2

        ene_lo, ene_hi = self.box_ene[:nene_ori], self.box_ene[1:]

        res = np.empty((nthetas, nene_ori))

        for i in range(nthetas):
            if thetas[i]<=15.:
                res[i, :] = calc_arf(theta=thetas[i],
                                     ebound_lo=ene_lo,
                                     ebound_hi=ene_hi,
                                     xmmsim=self)
            else:
                res[i, :] = 0.

        finterp = interp1d(thetas, res, axis=0)

        all_arfs = finterp(theta_image)

        # all_arfs = np.empty((self.box.shape[0], self.box.shape[1], nene_ori))
        #
        # for i in progressbar.progressbar(range(self.box.shape[0])):
        #
        #     for j in range(self.box.shape[1]):
        #
        #         if thetas[j, i] <= np.max(self.rads_vignetting):
        #
        #             all_arfs[j, i, :] = calc_arf(thetas[j, i], ene_lo, ene_hi, self)

        self.all_arfs = all_arfs

    def Pts(self, infile=None, outfile=None, outreg=None):
        '''
        Generate a point source list or read it from a previously loaded file

        :param infile: If provided, name of the input point source list to be loaded from a file.
        :type infile: str
        :param outfile: If provided, name of outpout file to save the point source list
        :type outfile: str
        :return:
        '''

        if infile is None and outfile is None:
            print('No infile or outfile provided, aborting')
            return

        if infile is not None and outfile is not None:
            print('Only one of infile and outfile can be provided, aborting')
            return

        if infile is not None and outfile is None:
            gen = False

        else:
            gen = True

        if gen:

            gen_sources(self,
                        outfile=outfile,
                        outreg=outreg)

            source_file = outfile

        else:

            source_file = infile

        pts_box(self,
                source_file=source_file)

        self.pts_file = source_file


    def ExtractImage(self, outname, elow=0.5, ehigh=2.0, nbin=10, withskybkg=True, withqpb=True, lhb=None, ght=None, ghn=None, cxb=None, NH=None, write_arf=False):
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

        print('# Generating exposure map...')
        expmap = exposure_map(self,
                              tsim=self.tsim,
                              elow=elow,
                              ehigh=ehigh,
                              nbin=nbin)

        if self.events:
            print('# Event file found, we will extract the image from the event file')

            print('# Extracting image from event file...')
            poisson_map = gen_image_evt(self,
                                        X_evt=self.X_evt,
                                        Y_evt=self.Y_evt,
                                        chan_evt=self.chan_evt,
                                        tsim=self.tsim,
                                        elow=elow,
                                        ehigh=ehigh,
                                        outfile=None)

            if withqpb:

                print('# Generating QPB map...')
                qpb_map = gen_qpb_image(self,
                                        tsim=self.tsim,
                                        elow=elow,
                                        ehigh=ehigh)

            else:
                qpb_map = poisson_map * 0.


        else:
            print('# No event file found, we will extract the map directly from the image box')

            print('# Generating box image...')
            box_map = gen_image_box(self,
                                    tsim=self.tsim,
                                    elow=elow,
                                    ehigh=ehigh,
                                    nbin=nbin)

            if not withskybkg:
                skybkg_map = box_map * 0.

            else:
                print('# Generating sky background map...')
                skybkg_map, expmap = gen_skybkg_image(self,
                                                      tsim=self.tsim,
                                                      elow=elow,
                                                      ehigh=ehigh,
                                                      nbin=nbin,
                                                      lhb=lhb,
                                                      ght=ght,
                                                      ghn=ghn,
                                                      cxb=cxb,
                                                      NH=NH,
                                                      abund=self.abund)


            if withqpb:

                print('# Generating QPB map...')
                qpb_map = gen_qpb_image(self,
                                        tsim=self.tsim,
                                        elow=elow,
                                        ehigh=ehigh)

            else:
                qpb_map = box_map * 0.

            print("# Simulating data...")
            poisson_map = np.random.poisson(box_map + qpb_map + skybkg_map)


        print('# Saving data into output files...')
        save_maps(self,
                  outname=outname,
                  countmap=poisson_map,
                  expmap=expmap,
                  bkgmap=qpb_map,
                  write_arf=write_arf)

    def ExtractFWC(self, calculate=False):
        """
        Determine the spectral shape of the FWC spectrum
        """
        if calculate:
            self.fwc_spec = gen_qpb_spectrum(self)

        else:
            area_tot = tot_area(self)

            self.fwc_spec = read_qpb_spectrum(self) * area_tot


    def ExtractSpectrum(self, outdir, cra, cdec, rin, rout, tsim_qpb=None, regfile=None, withskybkg=True,
                        withqpb=True, lhb=None, ght=None, ghn=None, cxb=None, NH=None):
        """
        Extract the spectrum, ARF and background file for an annulus between rin and rout. Regions can be masked by providing a DS9 region file.

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

        if self.events:
            print('# Event file found, we will extract the spectrum from the event file')

            print('# Extracting spectrum from event file...')
            spectrum, arf, backscal = gen_spec_evt(self,
                                                   cra=cra,
                                                   cdec=cdec,
                                                   rin=rin,
                                                   rout=rout,
                                                   regfile=regfile)

        else:
            print('# Extracting spectrum...')
            box_spec, arf, backscal = gen_spec_box(self,
                                                   tsim=self.tsim,
                                                   cra=cra,
                                                   cdec=cdec,
                                                   rin=rin,
                                                   rout=rout,
                                                   regfile=regfile)

            area_spec = backscal / 60. ** 2 * (0.05 ** 2)  # arcmin^2

            if withqpb:

                area_tot = tot_area(self)

                emin, emax = self.ebounds['E_MIN'], self.ebounds['E_MAX']

                bin_width = emax - emin

                if self.fwc_spec is None:
                    print('Please extract the FWC spectrum first')
                    qpb_spec = box_spec * 0.

                else:
                    print('# Extracting FWC spectrum...')

                    qpb_spec = np.random.poisson(self.fwc_spec * self.tsim * bin_width * area_spec / area_tot).astype(int)

            else:
                qpb_spec = box_spec * 0.

            if withskybkg:
                print('# Generating sky background spectrum...')
                skybkg_spec = gen_skybkg_spectrum(self,
                                                  tsim=self.tsim,
                                                  area_spec=area_spec,
                                                  arf=arf,
                                                  lhb=lhb,
                                                  ght=ght,
                                                  ghn=ghn,
                                                  cxb=cxb,
                                                  NH=NH,
                                                  abund=self.abund)
            else:
                skybkg_spec = box_spec * 0.

            spectrum = np.random.poisson(box_spec + skybkg_spec) + qpb_spec # Total spectrum

        if withqpb:

            area_spec = backscal / 60. ** 2 * (0.05 ** 2)  # arcmin^2

            area_tot = tot_area(self)

            emin, emax = self.ebounds['E_MIN'], self.ebounds['E_MAX']

            bin_width = emax - emin

            if tsim_qpb is None:

                tsim_qpb = self.tsim

            print('# Extracting FWC spectrum realization...')

            qpb_spec_out = np.random.poisson(self.fwc_spec * tsim_qpb * bin_width * area_spec / area_tot).astype(int)

        else:
            qpb_spec_out = None

        print('# Now saving spectra and region files...')
        save_spectrum(self,
                      outdir=outdir,
                      spectrum=spectrum,
                      tsim=self.tsim,
                      arf=arf,
                      qpb=qpb_spec_out,
                      backscal=backscal,
                      tsim_qpb=tsim_qpb)

    def ExtractSpectrumVoronoi(self, outdir, pixlist, tsim_qpb=None, regfile=None, withskybkg=True,
                               withqpb=True, lhb=None, ght=None, ghn=None, cxb=None, NH=None):
        """
        Extract the spectrum, ARF and background file for an annulus between rin and rout. Regions can be masked by providing a DS9 region file.

        :param outdir: Name of output directory
        :type outdir: str
        :param pixlist: Array with pixels chosen for spectral extraction
        :type pixlist: np array (N,2)
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

        if self.events:
            print('# Event file found, we will extract the spectrum from the event file')

            print('# Extracting spectrum from event file...')
            spectrum, arf, backscal = gen_spec_evt_pix(self, pixlist=pixlist)

        else:
            print('# You need to generate the event file first...terminating.')
            return
        if withqpb:

            area_spec = backscal / 60. ** 2 * (0.05 ** 2)  # arcmin^2

            area_tot = tot_area(self)

            emin, emax = self.ebounds['E_MIN'], self.ebounds['E_MAX']

            bin_width = emax - emin

            if tsim_qpb is None:
                tsim_qpb = self.tsim

            print('# Extracting FWC spectrum realization...')

            qpb_spec_out = np.random.poisson(self.fwc_spec * tsim_qpb * bin_width * area_spec / area_tot).astype(int)

        else:
            qpb_spec_out = None

        print('# Now saving spectra and region files...')
        save_spectrum(self,
                      outdir=outdir,
                      spectrum=spectrum,
                      tsim=self.tsim,
                      arf=arf,
                      qpb=qpb_spec_out,
                      backscal=backscal,
                      tsim_qpb=tsim_qpb)

    def ExtractEvents(self, outdir=None, withskybkg=True, withqpb=True, lhb=None, ght=None, ghn=None, cxb=None, NH=None):
        """
        Extract the spectrum, ARF and background file for an annulus between rin and rout. Regions can be masked by providing a DS9 region file.

        :param outdir: Name of output directory
        :type outdir: str
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
        if self.box:
            print('# Compute model box...')
            phot_box_ima = gen_phot_box(self,
                                        tsim=self.tsim,
                                        with_skybkg=withskybkg,
                                        lhb=lhb,
                                        ght=ght,
                                        ghn=ghn,
                                        cxb=cxb,
                                        NH=NH,
                                        abund=self.abund)

            print('# Generate sky events...')
            X_evt, Y_evt, chan_evt = gen_evt_list(self,
                                                  phot_box_ima=phot_box_ima)

        elif self.evtfile_input:
            X_evt, Y_evt, chan_evt = gen_phot_evtlist(self,
                                                      tsim=self.tsim,
                                                      with_skybkg=withskybkg,
                                                      lhb=lhb,
                                                      ght=ght,
                                                      ghn=ghn,
                                                      cxb=cxb,
                                                      NH=NH,
                                                      abund=self.abund)

        else:
            print('You need to provide either a box or an ideal event file.')
            return

        if self.fwc_spec is None:
            print('FWC events not extracted yet, skipping QPB event generation')

        if withqpb and self.fwc_spec is not None:
            print('# Generate QPB events...')
            X_qpb, Y_qpb, chan_qpb = gen_qpb_evt(self,
                                                 tsim=self.tsim)

            X_tot, Y_tot, chan_tot, time_tot = merge_evt((X_evt, X_qpb),
                                                         (Y_evt, Y_qpb),
                                                         (chan_evt, chan_qpb),
                                                         tsim=self.tsim)

        else:

            nevt = len(X_evt)

            itime = np.random.rand(nevt) * self.tsim

            args = np.argsort(itime)

            X_tot, Y_tot, chan_tot, time_tot = X_evt[args], Y_evt[args], chan_evt[args], itime[args]

        self.events = True
        self.X_evt = X_tot
        self.Y_evt = Y_tot
        self.chan_evt = chan_tot
        self.time_evt = time_tot

        if outdir is not None:

            if not os.path.exists(outdir):
                os.mkdir(outdir)

            save_evt_file(self,
                          X_evt=X_tot,
                          Y_evt=Y_tot,
                          chan_evt=chan_tot,
                          time_evt=time_tot,
                          tsim=self.tsim,
                          outfile=outdir+'/E'+self.instrument+'_events.fits')

    def LoadEvents(self, infile):
        '''
        Load events extracted from a previous run into the current session

        :param infile: Input file
        :type infile: str
        :return:
        '''

        print('# Reloading events from file '+infile)

        load_events(self,
                    infile=infile)




