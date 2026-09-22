# LSST–LISA Double White-Dwarf Candidate Analysis

This repository contains the analysis pipeline used to identify and characterize potential LSST optical counterparts to LISA double white-dwarf (DWD) sources. The analysis combines simulated LISA DWD catalogues with an LSST white-dwarf catalogue, evaluates source localization and optical properties, and applies logistic-regression models to assess candidate-selection performance.

## Input Catalogues

Place the input catalogues in the same directory as the analysis script.

### Required Files

- `LISA_source.csv`  
  Catalogue of simulated LISA double white-dwarf sources.

- `LSST_WD_24.csv`  
  LSST white-dwarf catalogue limited to apparent magnitude \(r < 24\).

### Optional Precomputed File

- `LISA_LSST_source.csv`  
  Precomputed LISA–LSST cross-match catalogue. When available, this file can include the `uncertainty_count` column, which stores the number of LSST white dwarfs within the localization region of each LISA source.

If this file is not available, the analysis calculates the required nearby-source counts from the LISA localization area and LSST sky positions.

## Required Catalogue Columns

### LISA Catalogue

The LISA source catalogue should provide the following columns:

```text
dist_new_kpc
delta_omega
chirp_masses
fmin
snr
galb
gall
```

where:

- `dist_new_kpc` is the source distance in kpc.
- `delta_omega` is the LISA sky-localization solid angle.
- `chirp_masses` is the binary chirp mass.
- `fmin` is the gravitational-wave frequency.
- `snr` is the LISA signal-to-noise ratio.
- `galb` and `gall` are Galactic latitude and longitude.

### LSST White-Dwarf Catalogue

The LSST catalogue should contain photometric, positional, and proper-motion information, including:

```text
umag
gmag
rmag
imag
zmag
ymag
l
b
pmdec
pmracosd
rmag_0, rmag_1, ..., rmag_99
```

The columns `rmag_0` through `rmag_99` are magnitude realizations used to evaluate the effect of photometric variation on candidate counts.

## Software Requirements

The analysis requires Python 3 and the following packages:

```text
numpy
pandas
matplotlib
seaborn
scipy
scikit-learn
astropy
healpy
```

They can be installed with:

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn astropy healpy
```

## Running the Analysis

Place the catalogues and the Python analysis script in the same directory, then run:

```bash
python lsst_lisa_dwd_analysis.py
```

Update the script filename in the command above if your local script uses a different name.

## Candidate Selection

The analysis identifies LSST white dwarfs located within the LISA localization regions. For each LISA source, the number of nearby LSST white dwarfs is denoted by `N_nearby`.

Two candidate samples are considered:

- **Low-confusion sample:** `N_nearby <= 10`
- **Intermediate-confusion sample:** `10 < N_nearby <= 100`

These samples are mutually exclusive. In particular, the intermediate-confusion sample does not include sources in the low-confusion sample.

The analysis also applies magnitude and proper-motion selection criteria. The default configuration uses:

```text
PM_LIMIT = 3.0
RMAG_LIMIT = 24.0
```

where `PM_LIMIT` is the proper-motion threshold and `RMAG_LIMIT` is the limiting LSST `r`-band magnitude.

## Sky Localization

LISA localization regions are calculated from the solid-angle uncertainty `delta_omega`. The equivalent angular radius, `delta_theta`, is derived from:

```text
delta_theta = arccos(1 - delta_omega / (2 pi))
```

The argument of `arccos` is clipped to the valid interval `[-1, 1]` to avoid numerical errors.

Nearby LSST sources are counted using a `BallTree` with the haversine metric, which is appropriate for angular separations on the sky.

## Photometric Realizations

The catalogue may include multiple `r`-band realizations:

```text
rmag_0, rmag_1, ..., rmag_99
```

The analysis evaluates candidate counts across these realizations to quantify the dependence of candidate selection on photometric variation.

The displayed realization is controlled by:

```python
DISPLAY_RMAG_COLUMN = "rmag_48"
```

The number of available realizations is specified by:

```python
N_RMAG_REALIZATIONS = 100
```

## Machine-Learning Analysis

Logistic-regression models are used to assess the separability of candidate populations.

Two feature configurations are evaluated.

### `mass_distance`

This model uses:

```text
chirp mass
distance
```

It tests how well the basic physical properties of a LISA DWD predict candidate selection.

### `log_all_parameters`

This model uses logarithmic physical parameters, including:

```text
chirp mass
distance
frequency
signal-to-noise ratio
sky-localization uncertainty
Galactic longitude
Galactic latitude
```

The precise set of variables is defined in the analysis script.

### Feature Scaling

`StandardScaler` is intentionally not applied.

The input variables retain their physical or logarithmic scales so that the logistic-regression coefficients remain physically interpretable.

### Model Evaluation

The analysis reports precision and recall for:

- The training data.
- The validation data.
- The full data set.

It also evaluates model performance over a range of probability thresholds:

```python
THRESHOLDS = np.linspace(0.01, 0.99, 100)
```

The default classification threshold is:

```python
THRESHOLD = 0.50
```

## Default Configuration

| Parameter | Default value | Description |
|---|---:|---|
| `PM_LIMIT` | `3.0` | Proper-motion selection limit |
| `RMAG_LIMIT` | `24.0` | LSST limiting `r`-band magnitude |
| `DISPLAY_RMAG_COLUMN` | `"rmag_48"` | Magnitude realization used in displayed figures |
| `N_RMAG_REALIZATIONS` | `100` | Number of `r`-band realizations |
| `N_NEARBY_VALUES` | `(10, 100)` | Nearby-source thresholds |
| `HEALPIX_BIN_SIZE_DEG` | `0.6` | Approximate HEALPix sky-bin size in degrees |
| `FIG_DPI` | `300` | Figure resolution |
| `THRESHOLD` | `0.50` | Logistic-regression classification threshold |
| `THRESHOLDS` | `0.01`–`0.99` | Threshold range for precision–recall curves |
| `DEGREE` | `2` | Polynomial-feature degree |
| `C_VALUE` | `1.0` | Logistic-regression inverse regularization strength |
| `TEST_SIZE` | `0.30` | Validation-set fraction |
| `N_REPEATS` | `100` | Number of repeated model evaluations |
| `RANDOM_STATE` | `42` | Random seed for reproducibility |

## Output Figures

The pipeline generates publication-quality PDF figures:

```text
LSST_WD.pdf
LSST_LISA_DWDs.pdf
candidate_count_distribution_N10.pdf
candidate_count_distribution_N100.pdf
SNR_rmag_and_uncertainties_WDs.pdf
distance_histogram.pdf
d_vs_M_candidates.pdf
pairwise_params.pdf
recall_precision_mass_distance.pdf
recall_precision_all_parameters.pdf
```

### Figure Descriptions

| File | Description |
|---|---|
| `LSST_WD.pdf` | HEALPix map of the LSST white-dwarf distribution. |
| `LSST_LISA_DWDs.pdf` | Sky distribution of LISA DWDs and LSST candidate samples. |
| `candidate_count_distribution_N10.pdf` | Candidate-count distribution for `N_nearby <= 10`. |
| `candidate_count_distribution_N100.pdf` | Candidate-count distribution for `10 < N_nearby <= 100`. |
| `SNR_rmag_and_uncertainties_WDs.pdf` | LISA signal-to-noise ratio, LSST magnitude, and localization diagnostics. |
| `distance_histogram.pdf` | Distance distributions for LISA DWDs and selected candidates. |
| `d_vs_M_candidates.pdf` | Chirp-mass versus distance distribution. |
| `pairwise_params.pdf` | Pairwise relationships among selected physical parameters. |
| `recall_precision_mass_distance.pdf` | Precision–recall results for the `mass_distance` model. |
| `recall_precision_all_parameters.pdf` | Precision–recall results for the `log_all_parameters` model. |

## Notes

- HEALPix map values represent the number of white dwarfs per HEALPix pixel. They are not sky number densities unless explicitly normalized by pixel area.
- Candidate markers are overlaid on relevant sky and parameter-space figures.
- The `N_nearby <= 10` and `10 < N_nearby <= 100` samples are separate, non-overlapping populations.
- All output figures are saved at the resolution specified by `FIG_DPI`.

## Reproducibility

For reproducible machine-learning splits and repeated analyses, the default random seed is:

```python
RANDOM_STATE = 42
```

Changing this value alters the train-validation split and may change the reported classification metrics.
