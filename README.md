LSST–LISA DWD Candidate Analysis
================================

Overview
--------
This script identifies and characterizes potential optical counterparts to LISA
double white dwarf (DWD) sources using an LSST white-dwarf catalogue.

The analysis:

1. Loads LISA DWD and LSST white-dwarf catalogues.
2. Applies a proper-motion cut to the LSST white-dwarf sample.
3. Counts LSST white dwarfs within each LISA sky-localization region.
4. Defines LISA–LSST candidate samples using:
      - LISA localization crowding, N_nearby;
      - an r-band magnitude limit;
      - LSST sky coverage.
5. Produces sky maps and diagnostic plots.
6. Quantifies variation across multiple r-band magnitude realizations.
7. Performs repeated logistic-regression validation using physically motivated
   LISA source parameters.

The script is intended to produce publication-quality figures and summary
statistics.


Required Input Files
--------------------
The analysis normally requires the following files in the working directory:

    LISA_source.csv
    LSST_WD_24.csv

An optional precomputed catalogue may also be used:

    LISA_LSST_source.csv

The precomputed catalogue is useful when the LSST white-dwarf catalogue is not
available or when recalculating the localization-region source counts is not
required.


Input Catalogue Requirements
----------------------------

LISA_source.csv
~~~~~~~~~~~~~~~
Required columns:

    umag
    gmag
    rmag
    imag
    zmag
    ymag
    l
    b
    dist_new_kpc
    delta_omega
    chirp_masses
    fmin
    snr
    rmag_0, rmag_1, ..., rmag_99

Column meanings:

    l
        Galactic longitude in degrees.

    b
        Galactic latitude in degrees.

    dist_new_kpc
        Heliocentric distance in kpc.

    delta_omega
        LISA sky-localization solid angle, assumed to be in steradians.

    chirp_masses
        Chirp mass in solar masses.

    fmin
        Initial or minimum gravitational-wave frequency in Hz.

    snr
        LISA signal-to-noise ratio.

    rmag_0 ... rmag_99
        Independent realizations of the LSST r-band magnitude.


LISA_LSST_source.csv
~~~~~~~~~~~~~~~~~~~~
Optional precomputed version of the LISA catalogue. It should contain all
columns required from LISA_source.csv, together with:

    uncertainty_count

Column meaning:

    uncertainty_count
        Number of LSST white dwarfs within the equivalent circular
        sky-localization region of each LISA source.

This file may be used to avoid recalculating BallTree localization counts.
Its row order must remain identical to that of the corresponding LISA source
catalogue.

Note that LSST_WD_24.csv is still required to reproduce the LSST sky map and
the HEALPix-based visibility mask.


LSST_WD_24.csv
~~~~~~~~~~~~~~
Required columns:

    galb
    gall
    pmdec
    pmracosd

Column meanings:

    galb
        Galactic latitude in degrees.

    gall
        Galactic longitude in degrees.

    pmdec
        Proper motion in declination.

    pmracosd
        Proper motion in right ascension multiplied by cos(declination).

The total proper motion is calculated as:

    pm = sqrt(pmdec^2 + pmracosd^2)


Software Requirements
---------------------
The code requires Python 3 and the following packages:

    numpy
    pandas
    matplotlib
    healpy
    scikit-learn
    tqdm

Recommended installation:

    pip install numpy pandas matplotlib healpy scikit-learn tqdm


Main Configuration Parameters
-----------------------------
The configuration section near the top of the script contains the main
analysis settings.

Input files:

    LISA_FILE = "LISA_source.csv"
    LSST_FILE = "LSST_WD_24.csv"

Selection thresholds:

    PM_LIMIT = 3.0
        Minimum LSST proper motion.

    RMAG_LIMIT = 24.0
        LSST r-band magnitude limit used for candidate selection.

    DISPLAY_RMAG_COLUMN = "rmag_48"
        Specific r-band realization used for displayed candidate plots.

    N_NEARBY_VALUES = (10, 100)
        Maximum allowed numbers of LSST white dwarfs within a LISA
        sky-localization region.

HEALPix map configuration:

    HEALPIX_BIN_SIZE_DEG = 0.6
        Approximate angular scale of the HEALPix sky map.

Figure output:

    FIG_DPI = 300
        Resolution used when saving figures.

Machine-learning configuration:

    THRESHOLD = 0.50
        Classification probability threshold.

    DEGREE = 2
        Polynomial degree used in logistic regression.

    C_VALUE = 1.0
        Inverse regularization strength in LogisticRegression.

    TEST_SIZE = 0.30
        Fraction of each realization reserved for validation.

    N_REPEATS = 100
        Number of repeated stratified train/test splits.

    N_RMAG_REALIZATIONS = 100
        Number of r-band magnitude realizations.


Candidate Definitions
---------------------

LSST visibility mask
~~~~~~~~~~~~~~~~~~~~
A LISA source is classified as visible when its HEALPix pixel contains at least
one LSST white dwarf after applying the proper-motion selection:

    visible_mask = pixel_counts[lisa_pixels] > 0


Localization crowding
~~~~~~~~~~~~~~~~~~~~~
For every LISA source, the script counts the number of LSST white dwarfs within
the equivalent circular LISA sky-localization region.

The angular radius is derived from the localization solid angle:

    delta_theta = arccos(1 - delta_omega / (2*pi))

where delta_omega is assumed to be in steradians.

The resulting number of nearby LSST white dwarfs is stored as:

    uncertainty_count

and is denoted in figures as:

    N_nearby


Candidate sample with N_nearby <= 10
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The low-crowding candidate sample is defined by:

    visible_mask
    AND uncertainty_count <= 10
    AND rmag <= 24


Candidate sample with 10 < N_nearby <= 100
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The intermediate-crowding candidate sample is defined by:

    visible_mask
    AND 10 < uncertainty_count <= 100
    AND rmag <= 24


Candidate sample with N_nearby <= 100
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This sample includes both the low- and intermediate-crowding candidates:

    visible_mask
    AND uncertainty_count <= 100
    AND rmag <= 24


Output Files
------------

LSST_WD.pdf
~~~~~~~~~~~
Aitoff projection of the LSST white-dwarf distribution after the proper-motion
cut. The colour scale shows the number of white dwarfs per HEALPix pixel.

LSST_LISA_DWDs.pdf
~~~~~~~~~~~~~~~~~~
Aitoff sky map of the LSST white-dwarf distribution with candidate LISA DWDs
overlaid.

Marker colours:

    Red:
        N_nearby <= 10

    Purple:
        10 < N_nearby <= 100


candidate_count_distribution_N10.pdf
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Histogram of the candidate count across the 100 r-band magnitude realizations
for the N_nearby <= 10 selection.

candidate_count_distribution_N100.pdf
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Histogram of the candidate count across the 100 r-band magnitude realizations
for the N_nearby <= 100 selection.


SNR_rmag_and_uncertainties_WDs.pdf
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Two-panel diagnostic figure:

    Panel (a):
        LISA SNR versus N_nearby.

    Panel (b):
        LISA SNR versus r-band magnitude for the N_nearby <= 10 and
        10 < N_nearby <= 100 samples.


distance_histogram.pdf
~~~~~~~~~~~~~~~~~~~~~~
Distance distributions for sources satisfying N_nearby <= 10 and
N_nearby <= 100.

The left panel has no magnitude cut. The right panel applies r <= 24.


d_vs_M_candidates.pdf
~~~~~~~~~~~~~~~~~~~~~
Heliocentric distance versus chirp mass for visible LISA sources and selected
LISA–LSST candidates.

The left panel shows the N_nearby <= 10 sample.

The right panel shows the N_nearby <= 100 sample.


pairwise_params.pdf
~~~~~~~~~~~~~~~~~~~
Pairwise distributions of LISA physical parameters:

    Heliocentric distance
    Localization area
    Gravitational-wave frequency
    Chirp mass

Grey points represent other visible sources. Red and purple points represent
the low- and intermediate-crowding candidate samples, respectively.


recall_precision_mass_distance.pdf
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Repeated-validation logistic-regression results using:

    Chirp mass
    Heliocentric distance

The model is trained separately for N_nearby <= 10 and N_nearby <= 100.


recall_precision_all_parameters.pdf
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Repeated-validation logistic-regression results using:

    log10(delta_omega)
    log10(fmin)
    log10(distance)
    log10(chirp mass)

The model predicts candidate membership among all visible sources.


Machine-Learning Analysis
-------------------------

Purpose
~~~~~~~
The logistic-regression analysis tests whether LISA source properties predict
membership in the optically detectable candidate sample.

Each r-band magnitude realization produces a separate binary classification
target:

    y = 1  if rmag_i <= 24
    y = 0  otherwise

within the applicable N_nearby selection.


Feature Sets
~~~~~~~~~~~~

1. mass_distance
----------------

Features:

    m = chirp_masses
    d = heliocentric distance

This model is evaluated only within the selected N_nearby sample.


2. log_all_parameters
---------------------

Features:

    log_delta_omega = log10(delta_omega)
    log_fmin        = log10(fmin)
    log_d           = log10(distance)
    log_chirp_mass  = log10(chirp mass)

This model is evaluated over all visible sources, while the target remains the
candidate selection for the relevant N_nearby threshold.


Polynomial Logistic Regression
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The model pipeline is:

    PolynomialFeatures(degree=2, include_bias=False)
    LogisticRegression(...)

No StandardScaler is used intentionally. This preserves the direct association
between fitted coefficients and the adopted physical variables or logarithmic
physical variables.

Because the features are not standardized, coefficient magnitudes should not
be compared directly between variables with different units or numerical
ranges without physical context.


Validation Procedure
~~~~~~~~~~~~~~~~~~~~
For every r-band magnitude realization:

    1. Stratified train/test splits are generated.
    2. The model is fit to the training subset.
    3. Predictions are evaluated for training, validation, and full samples.
    4. Precision, recall, ROC-AUC, average precision, and predicted source
       counts are recorded.
    5. Results are averaged over all successful fits.

The script uses:

    100 magnitude realizations
    100 repeated stratified splits per realization
    30% validation fraction

This corresponds to up to 10,000 fits for each N_nearby threshold and feature
set.


Notes and Assumptions
---------------------

1. The LISA localization area, delta_omega, is assumed to be expressed in
   steradians.

2. The equivalent circular localization radius is an approximation used to
   define a search region for LSST white dwarfs.

3. The HEALPix map is used to identify whether a source lies in a sky pixel
   containing at least one LSST white dwarf. This is not equivalent to a full
   LSST footprint or completeness model.

4. The label "White dwarfs per HEALPix pixel" represents source counts per
   HEALPix pixel, not a continuous surface density in units of deg^-2.

5. The rmag_0 to rmag_99 columns are treated as independent realizations of
   the apparent r-band magnitude.

6. If a candidate sample contains no sources for a given realization, the
   corresponding SNR fraction is recorded as NaN and ignored when calculating
   the mean fraction.

7. Logistic-regression performance can be sensitive to class imbalance,
   especially for rare candidate classes. Precision, recall, ROC-AUC, and
   average precision should therefore be interpreted together.


Suggested Execution
-------------------
Run the complete script from the directory containing the required catalogues:

    python lsst_lisa_dwd_analysis.py

The PDF figures are written to the working directory unless the output paths
are changed in the configuration section.


Citation and Acknowledgement
----------------------------
If this code is used in a publication, cite the relevant LISA, LSST/Rubin
Observatory, HEALPix, scikit-learn, NumPy, Pandas, Matplotlib, and BallTree
references, as appropriate for the data products and methods used.
