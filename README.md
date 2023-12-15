# XMM mock data simulator

A Python package to generate mock XMM-Newton images and spectra from 3D galaxy cluster simulations from hydrodynamical simulations. The tool generates particle background spectra from filter-wheel-closed data, a realistic sky background model, energy-dependent vignetting, energy redistribution, and PSF convolution.
## Installation

xmm_simulator depends on numpy, scipy, astropy, pyatomdb, and threeml. In particular, pyatomdb must be installed and the corresponding files must be stored in a directory linked with the _ATOMDB_ environment variable:

    export ATOMDB=/path/to/atomdb

To install xmm_simulator:

    git clone https://github.com/domeckert/xmm_simulator.git
    cd xmm_simulator
    pip install .

A directory containing the XMM calibration files (CCF) must also be available, see here:

https://www.cosmos.esa.int/web/xmm-newton/current-calibration-files

## Initialization

The xmm_simulator code is meant to work with data cubes containing a model spectrum in unit of photons/cm2/s/keV for each image pixel. The data cube must be of size N_ene x Npix_x x Npix_y , with N_ene the number of energy bins in the model spectrum, and Npix_x, Npix_y the number of input image pixels on the X and Y axes. The box size (in degrees) and the energy band definition must be provided by the user.

To initiate the code, do the following:

    import xmm_simulator
    xmmsim = xmm_simulator.XMMSimulator(boxfile='/path/to/data_cube',
                                        ccfpath='/path/to/ccf/',
                                        instrument='MOS1', # one constructor per instrument, can be 'MOS1', 'MOS2', or 'PN'
                                        tsim=25000, # simulation exposure time
                                        box_size=0.5, # size of provided image in degrees
                                        box_ene=None # numpy array containing the energy definition, i.e. it must have a size of N_ene+1 to contain the lower and upper boundaries of energy channels
                                        )

Once an _XMMSimulator_ object is defined, we can initiate the simulation by calculating a box of XMM ancillary response files (ARFs) for each point of the provided grid and for the provided instrument. This is done by reading the on-axis ARF and the vignetting curves at various energy bands from the CCF, and interpolating onto the chosen grid.

    xmmsim.ARF_Box()

The user can then optionally decide to include the non X-ray background (NXB) in the simulation by loading filter-wheel-closed (FWC) spectra. To create realistic simulations we recommend including the NXB. The NXB intensity is assumed to be flat over the detector, i.e. no soft proton treatment is implemented. 

    xmmsim.ExtractFWC(calculate=False)

If _calculate=False_ a pre-computed NXB spectrum is loaded (recommended). Otherwise, the NXB spectrum is calculated by reading the filter-wheel-closed event files provided in the CCF.

Finally, the user had the option of including randomly-positioned point sources in the simulation. The fluxes of the points sources are drawn from the logN-logS of Lehmer et al. (2013) and they are modeled as absorbed power laws, with the distribution of absorption column density (NH) flat in the range 20.5-23 and photon index drawn from a Gaussian with mean 1.9 and sigma 0.2 (Ueda et al. 2014).

    xmmsim.Pts(outfile='/path/to/output/file',
                infile=None,
                outreg='/path/to/output/region/file')

The user should provide only one of _infile_ and _outfile_. If _outfile_ is not None, a new source list is generated and stored into the provided output file. If _infile_ is provided, a previously extracted point source file is reloaded.

## Generating event files

We are now ready to extract a list of simulated events (position, energy and time) from the provided data cube

    xmmsim.ExtractEvents(outdir='/path/to/output/directory',
                        withskybkg=True, # Turn on/off sky background 
                        withqpb=True, # Turn on/off NXB
                        cxb=None, # Cosmic X-ray background norm per arcmin2, if None set to default value
                        lhb=None, # Local hot bubble norm per arcmin2, if None set to default value
                        ght=None, # Galactic halo temperature per arcmin2, if None set to default value
                        ghn=None, # Galactic halo norm per arcmin2, if None set to default value
                        NH=None # Galactic absorption column density (PhAbs model) in unit of 1e22 cm2, if None set to 0.05
                        )

The event file is stored into the output directory with the name of 'EMOS1_events.fits' (in the case of MOS1) and can be later reloaded using the LoadEvents tool:

    xmmsim.LoadEvents(infile='/path/to/event/file')

## Image extraction

Photon images, exposure maps, and NXB maps can be extracted from the generated event files by sorting the events into image pixels. This is done in the following way:

    xmmsim.ExtractImage(outname='outname', # name to be given to output files
                        elow=0.5, # Lower boundary of the energy band
                        ehigh=2.0, # Upper boundary of the energy band
                        write_arf=False # Set whether an on-axis ARF will be written or not
                        )

The code generates the following output files:

- _outname.fits_: count map
- _outname_expo.fits_: exposure map
- _outname_qpb.fits_: NXB map
- _outname.arf_: if write_arf=True, on-axis ARF

The maps generated from the three instruments (MOS1, MOS2, PN) can be summed to create combined EPIC maps with the _sum_maps_ tool,

    xmm_simulator.sum_maps(dir='/path/to/image/directory',
                            maps=('mos1S001.fits', 'mos2S002.fits', 'pnS003.fits'), 
                            instruments=('MOS1', 'MOS2', 'PN'), 
                            pnfact=3.42 # Ratio of PN to MOS ARF in the energy band of interest
                            )

The summed maps are in unit of MOS1 count rates

## Spectral extraction

Similarly, *xmm_simulator* can be used to extract spectra from the generated event files

    xmmsim.ExtractSpectrum(outdir='/path/to/output/directory',
                            cra=0., # RA of region center
                            cdec=0., # Dec of region center
                            rin=0., # Inner extraction radius in arcmin
                            rout=2.5, # Outer extraction radius in arcmin
                            tsim_qpb=None, # Exposure time of QPB spectrum if it is assumed to be extracted from a different data set (e.g. concatenated FWC event files). If None, the generated NXB spectrum has the same exposure time as the observation.
                            regfile=None # Region file containing a list of circular or elliptical regions to be masked
                            )

The tool extracts the following products:

- Count spectrum
- Redistribution matrix (RMF)
- Weighted ARF over the region of interest
- NXB spectrum

The extracted spectra can be readily loaded into XSPEC