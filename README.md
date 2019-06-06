# Improved detection of farside active regions using deep1learning analysis


## Introduction

The analysis of waves in the visible side of the Sun allows the detection of
active regions in the farside through local helioseismology
techniques. The knowledge of
the magnetism in the whole Sun, including the non-visible hemisphere, is
fundamental for several space weather forecasting
applications. During
the last years, the Solar TErrestrial Relationship Observatory
(STEREO) has been monitoring the farside of the Sun,
providing Extreme UltraViolet (EUV) images of that hemisphere. However, STEREO
spacecrafts are currently returning to the Earth-side of their orbit, and there
are no guaranties that they will be operative ten years from now, when they will
be back at the farside, as contact with STEREO-B is already lost. Thus, farside
helioseismology is the only available tool to obtain a continuous monitoring of
the non-visible solar hemisphere. Seismic identification of farside active
regions is challenged by the reduced signal-to-noise, and only large and strong
active regions can be reliable detected. Here we develop a new methodology to
improve the identification of active region signatures in farside seismic maps
using a deep learning approach. Our results show that this method can
significantly increase the number of detected farside active regions.


## Getting started

Once the dependencies are installed, the code is able to get farside probability
maps from farside phase-shift maps remapped onto a Carrington coordinate grid. 
Farside phase-shift maps computed from HMI Doppler maps and HMI magnetograms are
available through the JSOC (http://jsoc.stanford.edu/).

The code is simply run with:

    python farside_to_magnetogram.py -i INPUT -o OUTPUT [-b MAXBATCH] [-v VERBOSE]
    
where the parameters are:

- INPUT : IDL save file or HDF5 file that should contain a single dataset 
  with name `phases` of size [n_cases,11,nx,ny]. `n_cases` is often 1 but if
  many maps are available, they can be done in parallel. The number of time
  steps is 11, in steps of 12 hours, centered on the desired time.

- OUTPUT : HDF5 or FITS file with the resulting probability maps.

- MAXBATCH : maximum batch size, useful when running in GPUs with reduced amount
  of memory.

- VERBOSE : True/False for adding verbosity in the output

## Dependencies

    - pytorch (1.0 or later)
    - h5py
    - scipy
    - astropy
