"""
READ FIRST!!!

LSST–LISA DWD candidate selection, visualisation, and logistic-regression analysis.

The logistic-regression pipeline deliberately does not standardize features so that
the fitted coefficients retain their interpretation in the adopted physical units.

Don't forget to modify the data paths. 
LSST_WD_24.csv is the LSST WD catalog with r-band magnitude depth of 24 and is not provided.
Without LSST_WD_24.csv start uncomment # Load LISA_LSST_sources.csv and start from there.

The alternative machine learning scheme is commented out by default.
"""

import itertools
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import astropy.units as u
from matplotlib.ticker import FuncFormatter, LogLocator
from matplotlib.colors import LogNorm
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import BallTree
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from tqdm.auto import tqdm
from astropy.coordinates import CartesianDifferential, Galactocentric, Galactic, SkyCoord

# =============================================================================
# Configuration
# =============================================================================

LISA_FILE = "LISA_source.csv"
LSST_FILE = "LSST_WD_24.csv"

PM_LIMIT = 3.0
RMAG_LIMIT = 24.0
DISPLAY_RMAG_COLUMN = "rmag_48"
N_RMAG_REALIZATIONS = 100
N_NEARBY_VALUES = (10, 100)

HEALPIX_BIN_SIZE_DEG = 0.6
FIG_DPI = 300

# Logistic-regression configuration
THRESHOLD = 0.50
THRESHOLDS = np.linspace(0.01, 0.99, 100)
DEGREE = 2
C_VALUE = 1.0
TEST_SIZE = 0.30
N_REPEATS = 100
RANDOM_STATE = 42


# =============================================================================
# General utilities
# =============================================================================
def binsize_to_nside(bin_size_deg):
    """Return the nearest power-of-two HEALPix nside for a requested angular bin size."""
    bin_area_sr = np.deg2rad(bin_size_deg) ** 2
    nside_ideal = np.sqrt(np.pi / (3.0 * bin_area_sr))
    return 1 << int(np.round(np.log2(nside_ideal)))


def galactic_to_aitoff(longitude_deg, latitude_deg):
    """Convert Galactic longitude and latitude in degrees to Aitoff coordinates."""
    longitude = np.deg2rad(np.asarray(longitude_deg))
    latitude = np.deg2rad(np.asarray(latitude_deg))
    longitude = np.where(longitude > np.pi, longitude - 2.0 * np.pi, longitude)
    return -longitude, latitude


def save_figure(fig, filename):
    """Save a publication-quality figure."""
    fig.savefig(filename, dpi=FIG_DPI, bbox_inches="tight", pad_inches=0.02)


# =============================================================================
# Load and prepare catalogues
# =============================================================================
rmag_columns = [f"rmag_{index}" for index in range(N_RMAG_REALIZATIONS)]

lisa_columns = [
    "umag", "gmag", "rmag", "imag", "zmag", "ymag",
    "l", "b", "dist_new_kpc", "delta_omega", "chirp_masses", "fmin", "snr",
    *rmag_columns,
]

lisa = pd.read_csv(LISA_FILE, usecols=lisa_columns).rename(
    columns={
        "l": "gall",
        "b": "galb",
        "dist_new_kpc": "d",
        "snr": "SNR",
    }
)

if DISPLAY_RMAG_COLUMN not in lisa.columns:
    raise KeyError(f"{DISPLAY_RMAG_COLUMN!r} is not available in the LISA catalogue.")

# Convert localization solid angle to an equivalent circular angular radius.
lisa["delta_theta"] = np.degrees(
    np.arccos(np.clip(1.0 - lisa["delta_omega"] / (2.0 * np.pi), -1.0, 1.0))
)

lsst = pd.read_csv(LSST_FILE)
lsst["pm"] = np.hypot(lsst["pmdec"], lsst["pmracosd"])
lsst = lsst.loc[lsst["pm"] >= PM_LIMIT].reset_index(drop=True)

print(f"LSST white dwarfs with proper motion ≥ {PM_LIMIT:.0f}: {len(lsst):,}")


# =============================================================================
# Count LSST white dwarfs within each LISA localization region
# =============================================================================
# BallTree with the haversine metric expects [latitude, longitude] in radians.
lsst_coordinates = np.deg2rad(lsst[["galb", "gall"]].to_numpy())
lisa_coordinates = np.deg2rad(lisa[["galb", "gall"]].to_numpy())

tree = BallTree(lsst_coordinates, metric="haversine")
search_radii = np.deg2rad(lisa["delta_theta"].to_numpy())

lisa["uncertainty_count"] = tree.query_radius(
    lisa_coordinates,
    r=search_radii,
    count_only=True,
)

print(
    "Median number of LSST white dwarfs within a LISA localization region: "
    f"{lisa['uncertainty_count'].median():,.0f}"
)


# =============================================================================
# Load LISA_LSST_sources.csv
# =============================================================================
# lisa = pd.read_csv('LISA_LSST_sources.csv')


# =============================================================================
# Construct HEALPix sky map
# =============================================================================
nside = binsize_to_nside(HEALPIX_BIN_SIZE_DEG)
npix = hp.nside2npix(nside)

theta_lsst = np.deg2rad(90.0 - lsst["galb"].to_numpy())
phi_lsst = np.deg2rad(lsst["gall"].to_numpy())
lsst_pixels = hp.ang2pix(nside, theta_lsst, phi_lsst)

pixel_counts = np.bincount(lsst_pixels, minlength=npix).astype(float)

resolution_rad = hp.nside2resol(nside)
nlon = int(2.0 * np.pi / resolution_rad)
nlat = int(np.pi / resolution_rad)

lon_grid = np.linspace(-np.pi, np.pi, nlon)
lat_grid = np.linspace(-np.pi / 2.0, np.pi / 2.0, nlat)
lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

theta_mesh = np.pi / 2.0 - lat_mesh
phi_mesh = -lon_mesh
map_pixels = hp.ang2pix(nside, theta_mesh, phi_mesh)

sky_map = pixel_counts[map_pixels]
sky_map[sky_map == 0] = np.nan

theta_lisa = np.deg2rad(90.0 - lisa["galb"].to_numpy())
phi_lisa = np.deg2rad(lisa["gall"].to_numpy())
lisa_pixels = hp.ang2pix(nside, theta_lisa, phi_lisa)

# A source is considered visible if its sky pixel contains at least one LSST WD.
visible_mask = pixel_counts[lisa_pixels] > 0
magnitude_mask = lisa[DISPLAY_RMAG_COLUMN].le(RMAG_LIMIT)


# =============================================================================
# LSST white-dwarf sky map
# =============================================================================
fig, ax = plt.subplots(figsize=(8.0, 5.5), subplot_kw={"projection": "aitoff"})

image = ax.pcolormesh(
    lon_mesh,
    lat_mesh,
    sky_map,
    cmap="viridis",
    norm=LogNorm(vmin=1, vmax=np.nanmax(sky_map)),
    rasterized=True,
)

xticks = np.arange(-150, 151, 30)
yticks = np.arange(-75, 76, 15)

ax.set_xticks(np.deg2rad(xticks))
ax.set_xticklabels([f"{value}°" for value in xticks[::-1]], fontsize=8)
ax.set_yticks(np.deg2rad(yticks))
ax.set_yticklabels([f"{value}°" for value in yticks], fontsize=8)

ax.grid(alpha=0.35, linestyle="--", linewidth=0.5)
ax.set_title("LSST white dwarfs", pad=18)

colorbar = fig.colorbar(image, ax=ax, pad=0.02, shrink=0.65, aspect=18)
colorbar.set_label("White dwarfs per HEALPix pixel")

save_figure(fig, "LSST_WD.pdf")
plt.show()


# =============================================================================
# Candidate masks
# =============================================================================
sky_mask_10 = visible_mask & lisa["uncertainty_count"].le(10)
sky_mask_100 = visible_mask & lisa["uncertainty_count"].le(100)
sky_mask_10_100 = visible_mask & lisa["uncertainty_count"].between(11, 100)

candidate_mask_10 = sky_mask_10 & magnitude_mask
candidate_mask_100 = sky_mask_100 & magnitude_mask
candidate_mask_10_100 = sky_mask_10_100 & magnitude_mask

print(f"N_nearby ≤  10: {int(candidate_mask_10.sum()):,} candidates; {100.0 * lisa.loc[candidate_mask_10, 'SNR'].ge(100).mean():.0f}% with SNR ≥ 100.")
print(f"N_nearby ≤  100: {int(candidate_mask_100.sum()):,} candidates; {100.0 * lisa.loc[candidate_mask_100, 'SNR'].ge(100).mean():.0f}% with SNR ≥ 100.")


# =============================================================================
# LSST sky map with LISA candidate overlay
# =============================================================================
fig, ax = plt.subplots(figsize=(8.0, 5.5), subplot_kw={"projection": "aitoff"})

image = ax.pcolormesh(
    lon_mesh,
    lat_mesh,
    sky_map,
    cmap="viridis",
    norm=LogNorm(vmin=1, vmax=np.nanmax(sky_map)),
    rasterized=True,
)

for mask, color, label in [
    (candidate_mask_10, "tab:red", r"$N_{\rm nearby} \leq 10$"),
    (candidate_mask_10_100, "tab:purple", r"$10 < N_{\rm nearby} \leq 100$"),
]:
    longitude, latitude = galactic_to_aitoff(
        lisa.loc[mask, "gall"],
        lisa.loc[mask, "galb"],
    )
    ax.scatter(
        longitude,
        latitude,
        s=45,
        marker="*",
        color=color,
        edgecolor="white",
        linewidth=0.25,
        label=label,
        zorder=3,
    )

ax.set_xticks(np.deg2rad(xticks))
ax.set_xticklabels([f"{value}°" for value in xticks[::-1]], fontsize=8)
ax.set_yticks(np.deg2rad(yticks))
ax.set_yticklabels([f"{value}°" for value in yticks], fontsize=8)

ax.grid(alpha=0.35, linestyle="--", linewidth=0.5)
ax.set_title("LSST–LISA DWD candidates", pad=18)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False, markerscale=2)

colorbar = fig.colorbar(image, ax=ax, pad=0.02, shrink=0.65, aspect=18)
colorbar.set_label("White dwarfs per HEALPix pixel")

save_figure(fig, "LSST_LISA_DWDs.pdf")
plt.show()


# =============================================================================
# Candidate statistics across r-band realizations
# =============================================================================
for n_nearby in N_NEARBY_VALUES:
    selection_mask = visible_mask & lisa["uncertainty_count"].le(n_nearby)

    candidate_counts = []
    high_snr_fractions = []

    for rmag_column in rmag_columns:
        candidate_mask = selection_mask & lisa[rmag_column].le(RMAG_LIMIT)
        n_candidates = candidate_mask.sum()

        candidate_counts.append(n_candidates)
        high_snr_fractions.append(
            lisa.loc[candidate_mask, "SNR"].ge(100).mean() if n_candidates else np.nan
        )

    candidate_counts = np.asarray(candidate_counts)
    high_snr_fractions = np.asarray(high_snr_fractions)

    print(f"\nN_nearby ≤ {n_nearby}")
    print(f"Mean candidate count: {candidate_counts.mean():.0f} ± {candidate_counts.std():.0f}")
    print(
        "Mean fraction with SNR ≥ 100: "
        f"{100.0 * np.nanmean(high_snr_fractions):.0f} ± "
        f"{100.0 * np.nanstd(high_snr_fractions):.0f}%"
    )

    fig, ax = plt.subplots(figsize=(5.0, 3.8))
    ax.hist(candidate_counts, bins="auto", color="steelblue", edgecolor="white")
    ax.axvline(candidate_counts.mean(), color="crimson", linestyle="--", linewidth=1.2)
    ax.set_xlabel(r"$N_{\rm cand}$")
    ax.set_ylabel("Number of realizations")
    ax.set_title(rf"$N_{{\rm nearby}} \leq {n_nearby}$")
    ax.grid(alpha=0.25, linestyle=":")
    save_figure(fig, f"candidate_count_distribution_N{n_nearby}.pdf")
    plt.show()


# =============================================================================
# SNR versus localization and r-band magnitude
# =============================================================================
fig, axes = plt.subplots(
    1,
    2,
    figsize=(12.5, 5.2),
    subplot_kw={"box_aspect": 1},
)

axes[0].scatter(
    lisa.loc[visible_mask, "uncertainty_count"],
    lisa.loc[visible_mask, "SNR"],
    s=8,
    alpha=0.25,
    linewidths=0,
    rasterized=True,
)

axes[0].axvline(10, color="tab:red", linestyle="--", label=r"$N_{\rm nearby}=10$")
axes[0].axvline(100, color="tab:red", linestyle=":", label=r"$N_{\rm nearby}=100$")
axes[0].set_xscale("symlog", linthresh=1)
axes[0].set_xlim(left=-0.1)
axes[0].set_yscale("log")
axes[0].set_xlabel(r"$N_{\rm nearby}$")
axes[0].set_ylabel("SNR")
axes[0].set_title("(a)", loc="left", pad=4)
axes[0].grid(alpha=0.30, linestyle=":")
axes[0].legend(frameon=False)

axes[1].scatter(
    lisa.loc[sky_mask_10_100, DISPLAY_RMAG_COLUMN],
    lisa.loc[sky_mask_10_100, "SNR"],
    s=8,
    alpha=0.30,
    linewidths=0,
    color="tab:purple",
    label=r"$10 < N_{\rm nearby} \leq 100$",
    rasterized=True,
)

axes[1].scatter(
    lisa.loc[sky_mask_10, DISPLAY_RMAG_COLUMN],
    lisa.loc[sky_mask_10, "SNR"],
    s=8,
    alpha=0.4,
    linewidths=0,
    color="tab:orange",
    label=r"$N_{\rm nearby} \leq 10$",
    rasterized=True,
)

axes[1].axvline(RMAG_LIMIT, color="tab:green", linestyle="--", label=r'$r_{\rm mag}=$'+f'{RMAG_LIMIT:.0f}')
axes[1].set_yscale("log")
axes[1].set_xlabel(r"$r_{\rm mag}$")
axes[1].set_ylabel("SNR")
axes[1].set_title("(b)", loc="left", pad=4)
axes[1].grid(alpha=0.30, linestyle=":")
axes[1].legend(frameon=False, markerscale=2)

fig.tight_layout()
save_figure(fig, "SNR_rmag_and_uncertainties_WDs.pdf")
plt.show()


# =============================================================================
# Distance distributions
# =============================================================================
distance_10_all = lisa.loc[sky_mask_10, "d"]
distance_100_all = lisa.loc[sky_mask_100, "d"]

distance_10_mag = lisa.loc[candidate_mask_10, "d"]
distance_100_mag = lisa.loc[candidate_mask_100, "d"]

bins_all = np.linspace(distance_100_all.min(), distance_100_all.max(), 40)
bins_mag = np.linspace(distance_100_mag.min(), distance_100_mag.max(), 25)

fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))

axes[0].hist(
    distance_100_all,
    bins=bins_all,
    color="steelblue",
    alpha=0.65,
    edgecolor="white",
    label=r"$N_{\rm nearby} \leq 100$",
)

axes[0].hist(
    distance_10_all,
    bins=bins_all,
    color="crimson",
    alpha=0.70,
    edgecolor="white",
    label=r"$N_{\rm nearby} \leq 10$",
)

axes[0].set_xlabel("Heliocentric distance (kpc)")
axes[0].set_ylabel(r"$N_{\rm cand}$")
axes[0].set_title("(a) No magnitude cut", loc="left", pad=4)
axes[0].legend(frameon=False)
axes[0].grid(alpha=0.25, linestyle=":")

axes[1].hist(
    distance_100_mag,
    bins=bins_mag,
    color="steelblue",
    alpha=0.65,
    edgecolor="white",
    label=r"$N_{\rm nearby} \leq 100$",
)

axes[1].hist(
    distance_10_mag,
    bins=bins_mag,
    color="crimson",
    alpha=0.70,
    edgecolor="white",
    label=r"$N_{\rm nearby} \leq 10$",
)

axes[1].set_xlabel("Heliocentric distance (kpc)")
axes[1].set_ylabel(r"$N_{\rm cand}$")
axes[1].set_title(r"(b) $r_{\rm mag} \leq$"+f"{RMAG_LIMIT:.0f}", loc="left", pad=4)
axes[1].legend(frameon=False)
axes[1].grid(alpha=0.25, linestyle=":")

fig.tight_layout()
save_figure(fig, "distance_histogram.pdf")
plt.show()


# =============================================================================
# Distance versus chirp mass
# =============================================================================
fig, axes = plt.subplots(
    1,
    2,
    figsize=(12.5, 5.0),
    subplot_kw={"box_aspect": 1},
)

for ax, sky_mask, candidate_mask, title in [
    (axes[0], sky_mask_10, candidate_mask_10, r"$N_{\rm nearby} \leq 10$"),
    (axes[1], sky_mask_100, candidate_mask_100, r"$N_{\rm nearby} \leq 100$"),
]:
    ax.scatter(
        lisa.loc[sky_mask, "d"],
        lisa.loc[sky_mask, "chirp_masses"],
        s=8,
        alpha=0.25,
        linewidths=0,
        color="0.45",
        rasterized=True,
        label="Dim sources",
    )

    ax.scatter(
        lisa.loc[candidate_mask, "d"],
        lisa.loc[candidate_mask, "chirp_masses"],
        s=10,
        alpha=0.70,
        linewidths=0,
        color="tab:orange",
        rasterized=True,
        label="Candidates",
    )

    ax.set_xlabel("Heliocentric distance (kpc)")
    ax.set_ylabel(r"Chirp mass ($M_\odot$)")
    ax.set_title(title, pad=4)
    ax.grid(alpha=0.30, linestyle=":")
    ax.legend(frameon=False, markerscale=2)

fig.tight_layout()
save_figure(fig, "d_vs_M_candidates.pdf")
plt.show()


# =============================================================================
# Pairwise LISA parameter distributions
# =============================================================================
def pairwise_values(mask, column):
    values = lisa.loc[mask, column]
    return values / 0.1 if column == "chirp_masses" else values

lisa["delta_omega_deg2"] = lisa["delta_omega"] * (180.0 / np.pi) ** 2
lisa["chirp_masses_10"] = lisa["chirp_masses"] * 10
parameters = ["d", "delta_omega_deg2", "fmin", "chirp_masses_10"]
parameter_labels = {
    "d": "Heliocentric distance (kpc)",
    "delta_omega_deg2": r"$\Delta\Omega$ (deg$^2$)",
    "fmin": r"$f_0$ (Hz)",
    "chirp_masses_10": r"Chirp mass ($10^{-1}M_\odot$)",
}

background_mask = visible_mask & ~(candidate_mask_10 | candidate_mask_10_100)
pairs = list(itertools.combinations(parameters, 2))

ncols = 3
nrows = int(np.ceil(len(pairs) / ncols))

log_tick_formatter = FuncFormatter(
    lambda value, _: f"{value:g}" if value > 0 else ""
)

fig, axes = plt.subplots(
    nrows,
    ncols,
    figsize=(4.2 * ncols, 4.0 * nrows),
    subplot_kw={"box_aspect": 1},
    squeeze=False,
)

for index, (x_name, y_name) in enumerate(pairs):
    ax = axes.flat[index]

    ax.scatter(
        lisa.loc[background_mask, x_name],
        lisa.loc[background_mask, y_name],
        s=6,
        alpha=0.20,
        color="0.5",
        linewidths=0,
        rasterized=True,
        label="Dim or clustered sources",
    )

    ax.scatter(
        lisa.loc[candidate_mask_10, x_name],
        lisa.loc[candidate_mask_10, y_name],
        s=6,
        alpha=0.70,
        color="tab:orange",
        linewidths=0,
        rasterized=True,
        label=r"$N_{\rm nearby} \leq 10$",
    )

    ax.scatter(
        lisa.loc[candidate_mask_10_100, x_name],
        lisa.loc[candidate_mask_10_100, y_name],
        s=6,
        alpha=0.70,
        color="tab:purple",
        linewidths=0,
        rasterized=True,
        label=r"$10 < N_{\rm nearby} \leq 100$",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")

    if x_name == "chirp_masses_10":
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0,)))
        ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10,2)))
        ax.xaxis.set_major_formatter(log_tick_formatter)
        ax.xaxis.set_minor_formatter(log_tick_formatter)
        ax.tick_params(axis="x", which="minor", labelsize=9)
    if y_name == "chirp_masses_10":
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0,)))
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10,2)))
        ax.yaxis.set_major_formatter(log_tick_formatter)
        ax.yaxis.set_minor_formatter(log_tick_formatter)
        ax.tick_params(axis="y", which="minor", labelsize=9)
 
    ax.set_xlabel(parameter_labels[x_name])   
    ax.set_ylabel(parameter_labels[y_name])    
    ax.set_title(f"({chr(97 + index)})", loc="left", pad=4)
    ax.grid(alpha=0.25, linestyle=":")

for ax in axes.flat[len(pairs):]:
    ax.remove()

handles, labels = axes.flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, markerscale=3)
fig.subplots_adjust(top=0.89, hspace=0.35, wspace=0.35)

save_figure(fig, "pairwise_params.pdf")
plt.show()


# =============================================================================
# Machine-learning helper functions
# =============================================================================
def curves_from_scores(y_true, y_scores, thresholds):
    """Return predicted count, recall, and precision over probability thresholds."""
    y_true = np.asarray(y_true, dtype=int)
    y_scores = np.asarray(y_scores, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)

    valid = np.isfinite(y_scores)
    y_true = y_true[valid]
    y_scores = y_scores[valid]

    if len(y_true) == 0:
        zeros = np.zeros(len(thresholds))
        return zeros, zeros, zeros

    order = np.argsort(y_scores)
    scores = y_scores[order]
    labels = y_true[order]

    n_total = len(labels)
    n_positive = labels.sum()

    cumulative_positive = np.concatenate([[0], np.cumsum(labels)])
    start_indices = np.searchsorted(scores, thresholds, side="left")

    n_predicted = n_total - start_indices
    true_positive = n_positive - cumulative_positive[start_indices]

    precision = np.divide(
        true_positive,
        n_predicted,
        out=np.zeros_like(true_positive, dtype=float),
        where=n_predicted > 0,
    )

    recall = np.divide(
        true_positive,
        n_positive,
        out=np.zeros_like(true_positive, dtype=float),
        where=n_positive > 0,
    )

    return n_predicted, recall, precision


def mean_std(sum_values, sumsq_values, n_values):
    """Compute mean and standard deviation from accumulated sums."""
    mean = sum_values / n_values
    std = np.sqrt(np.maximum(sumsq_values / n_values - mean**2, 0.0))
    return mean, std


def set_safe_ylim(ax, lower, upper, minimum_top=None, padding=0.05):
    """Set non-negative y limits with a small upper margin."""
    ymin = np.nanmin(lower)
    ymax = np.nanmax(upper)
    span = ymax - ymin

    ymax += padding * (span if span > 0 else max(abs(ymax), 1.0))

    if minimum_top is not None:
        ymax = max(ymax, minimum_top)

    ax.set_ylim(max(0.0, ymin), ymax)


def build_features(data, feature_set):
    """
    Construct physically interpretable feature sets.

    No StandardScaler is used intentionally: coefficients remain tied to the
    adopted physical or logarithmic physical variables.
    """
    if feature_set == "mass_distance":
        return pd.DataFrame(
            {
                "m": data["chirp_masses"],
                "d": data["d"],
            }
        )

    if feature_set == "log_all_parameters":
        return pd.DataFrame(
            {
                "log_delta_omega": np.log10(data["delta_omega"].clip(lower=1e-12)),
                "log_fmin": np.log10(data["fmin"].clip(lower=1e-12)),
                "log_d": np.log10(data["d"].clip(lower=1e-12)),
                "log_chirp_mass": np.log10(data["chirp_masses"].clip(lower=1e-12)),
            }
        )

    raise ValueError(f"Unknown feature set: {feature_set}")


def make_target(data, selection_mask, rmag_column):
    """Create binary labels for one r-band magnitude realization."""
    target_mask = selection_mask & data[rmag_column].le(RMAG_LIMIT)
    labels = target_mask.loc[selection_mask].astype(int).reset_index(drop=True)
    return labels, target_mask


def repeated_validation(model, X, y_by_rmag):
    """Perform repeated stratified validation for all magnitude realizations."""
    curve_names = [
        "train_npred", "train_recall", "train_precision",
        "test_npred", "test_recall", "test_precision",
        "all_npred", "all_recall", "all_precision",
    ]

    n_thresholds = len(THRESHOLDS)
    total_fits = N_REPEATS * len(y_by_rmag)

    curve_sums = {name: np.zeros(n_thresholds) for name in curve_names}
    curve_sumsq = {name: np.zeros(n_thresholds) for name in curve_names}

    metrics = {
        "split_id": [],
        "rmag_id": [],
        "precision": [],
        "recall": [],
        "roc_auc": [],
        "average_precision": [],
        "n_positive_train": [],
        "n_positive_test": [],
        "n_predicted_test": [],
    }

    validation_labels = []
    validation_scores = []
    n_successful_fits = 0

    progress = tqdm(
        total=total_fits,
        desc="Repeated validation",
        unit="fit",
        dynamic_ncols=True,
        mininterval=1.0,
    )

    try:
        for rmag_id, y in enumerate(y_by_rmag):
            splitter = StratifiedShuffleSplit(
                n_splits=N_REPEATS,
                test_size=TEST_SIZE,
                random_state=RANDOM_STATE + rmag_id,
            )

            for split_id, (train_index, test_index) in enumerate(splitter.split(X, y)):
                progress.set_postfix(
                    split=f"{split_id + 1}/{N_REPEATS}",
                    rmag=f"{rmag_id + 1}/{len(y_by_rmag)}",
                )

                X_train = X.iloc[train_index]
                X_test = X.iloc[test_index]
                y_train = y.iloc[train_index]
                y_test = y.iloc[test_index]

                if y_train.nunique() < 2 or y_test.nunique() < 2:
                    progress.update(1)
                    continue

                fitted_model = clone(model)
                fitted_model.fit(X_train, y_train)

                train_scores = fitted_model.predict_proba(X_train)[:, 1]
                test_scores = fitted_model.predict_proba(X_test)[:, 1]
                all_scores = fitted_model.predict_proba(X)[:, 1]

                test_prediction = (test_scores >= THRESHOLD).astype(int)

                metrics["split_id"].append(split_id)
                metrics["rmag_id"].append(rmag_id)
                metrics["precision"].append(
                    precision_score(y_test, test_prediction, zero_division=0)
                )
                metrics["recall"].append(
                    recall_score(y_test, test_prediction, zero_division=0)
                )
                metrics["roc_auc"].append(roc_auc_score(y_test, test_scores))
                metrics["average_precision"].append(
                    average_precision_score(y_test, test_scores)
                )
                metrics["n_positive_train"].append(y_train.sum())
                metrics["n_positive_test"].append(y_test.sum())
                metrics["n_predicted_test"].append(test_prediction.sum())

                curves = {
                    "train_npred": curves_from_scores(y_train, train_scores, THRESHOLDS)[0],
                    "train_recall": curves_from_scores(y_train, train_scores, THRESHOLDS)[1],
                    "train_precision": curves_from_scores(y_train, train_scores, THRESHOLDS)[2],
                    "test_npred": curves_from_scores(y_test, test_scores, THRESHOLDS)[0],
                    "test_recall": curves_from_scores(y_test, test_scores, THRESHOLDS)[1],
                    "test_precision": curves_from_scores(y_test, test_scores, THRESHOLDS)[2],
                    "all_npred": curves_from_scores(y, all_scores, THRESHOLDS)[0],
                    "all_recall": curves_from_scores(y, all_scores, THRESHOLDS)[1],
                    "all_precision": curves_from_scores(y, all_scores, THRESHOLDS)[2],
                }

                for name, values in curves.items():
                    curve_sums[name] += values
                    curve_sumsq[name] += values**2

                validation_labels.append(y_test.to_numpy())
                validation_scores.append(test_scores)

                n_successful_fits += 1
                progress.update(1)

    finally:
        progress.close()

    if n_successful_fits == 0:
        raise RuntimeError("No successful logistic-regression fits were produced.")

    curve_summary = {
        "thresholds": THRESHOLDS,
        "n_successful_fits": n_successful_fits,
    }

    for name in curve_names:
        mean, std = mean_std(curve_sums[name], curve_sumsq[name], n_successful_fits)
        curve_summary[f"{name}_mean"] = mean
        curve_summary[f"{name}_std"] = std

    metrics = {name: np.asarray(values) for name, values in metrics.items()}

    return (
        np.concatenate(validation_labels),
        np.concatenate(validation_scores),
        metrics,
        curve_summary,
    )


def print_logistic_equation(model, feature_columns, n_nearby, feature_set):
    """Print the fitted polynomial logistic-regression equation."""
    polynomial = model.named_steps["polynomialfeatures"]
    classifier = model.named_steps["logisticregression"]

    feature_names = polynomial.get_feature_names_out(feature_columns)
    coefficients = classifier.coef_[0]

    terms = [f"({classifier.intercept_[0]:.2f})"]

    for name, coefficient in zip(feature_names, coefficients):
        sign = "+" if coefficient >= 0 else "-"
        terms.append(f" {sign} ({abs(coefficient):.2f} × {name})")

    print("\n===== Final Logistic Regression Equation =====")
    print(f"Feature set: {feature_set}")
    print(f"Degree: {DEGREE}")
    print(f"C: {C_VALUE}")
    print(f"N_nearby ≤ {n_nearby}")
    print("logit(p) = " + "".join(terms))
    print("p = 1 / [1 + exp(-logit(p))]")


def plot_ml_case(n_nearby, axes, curve_summary, mean_target_count, std_target_count):
    """Plot recall versus selected count and recall versus precision."""
    ax_count, ax_precision = axes
    thresholds = curve_summary["thresholds"]
    threshold_index = np.argmin(np.abs(thresholds - THRESHOLD))

    all_recall = curve_summary["all_recall_mean"]
    all_count = curve_summary["all_npred_mean"]
    all_count_std = curve_summary["all_npred_std"]

    all_count_low = np.maximum(all_count - all_count_std, 0)
    all_count_high = all_count + all_count_std

    ax_count.fill_between(
        all_recall,
        all_count_low,
        all_count_high,
        color="tab:orange",
        alpha=0.25,
        linewidth=0,
        label=r"All-data $1\sigma$",
    )

    ax_count.plot(
        all_recall,
        all_count,
        color="tab:orange",
        linewidth=1.2,
        label="Mean all-data curve",
    )

    ax_count.scatter(
        all_recall[threshold_index],
        all_count[threshold_index],
        color="crimson",
        marker="*",
        s=100,
        zorder=5,
        label=rf"$p={THRESHOLD:.1f}$",
    )

    ax_count.set_title(
        rf"$N_{{\rm nearby}} \leq {n_nearby}$, "
        rf"$N_{{\rm cand}}={mean_target_count:.0f}\pm{std_target_count:.0f}$"
    )

    ax_count.set_xlabel("Recall")
    ax_count.set_ylabel(r"$N_{\rm selected}$")
    ax_count.set_xlim(-0.05, 1.05)
    ax_count.grid(alpha=0.25, linestyle=":")
    ax_count.legend(fontsize=8, frameon=False)
    ax_count.set_box_aspect(1)

    set_safe_ylim(ax_count, all_count_low, all_count_high)

    test_recall = curve_summary["test_recall_mean"]
    test_precision = curve_summary["test_precision_mean"]
    test_precision_std = curve_summary["test_precision_std"]

    train_recall = curve_summary["train_recall_mean"]
    train_precision = curve_summary["train_precision_mean"]
    train_precision_std = curve_summary["train_precision_std"]

    test_low = np.maximum(test_precision - test_precision_std, 0)
    test_high = test_precision + test_precision_std

    train_low = np.maximum(train_precision - train_precision_std, 0)
    train_high = train_precision + train_precision_std

    ax_precision.fill_between(
        test_recall,
        test_low,
        test_high,
        color="tab:orange",
        alpha=0.25,
        linewidth=0,
        label=r"Validation $1\sigma$",
    )

    ax_precision.plot(
        test_recall,
        test_precision,
        color="tab:orange",
        linewidth=1.2,
        label="Validation",
    )

    ax_precision.fill_between(
        train_recall,
        train_low,
        train_high,
        color="tab:blue",
        alpha=0.20,
        linewidth=0,
        label=r"Training $1\sigma$",
    )

    ax_precision.plot(
        train_recall,
        train_precision,
        color="tab:blue",
        linewidth=1.2,
        label="Training",
    )

    ax_precision.scatter(
        test_recall[threshold_index],
        test_precision[threshold_index],
        color="crimson",
        marker="*",
        s=100,
        zorder=5,
        label=rf"$p={THRESHOLD:.1f}$",
    )

    ax_precision.set_title(rf"$N_{{\rm nearby}} \leq {n_nearby}$")
    ax_precision.set_xlabel("Recall")
    ax_precision.set_ylabel("Precision")
    ax_precision.set_xlim(-0.05, 1.05)
    ax_precision.grid(alpha=0.25, linestyle=":")
    ax_precision.legend(fontsize=8, frameon=False)
    ax_precision.set_box_aspect(1)

    set_safe_ylim(
        ax_precision,
        np.concatenate([test_low, train_low]),
        np.concatenate([test_high, train_high]),
        minimum_top=1.0,
    )


def run_ml_case(data, n_nearby, feature_set, restrict_input_to_selection):
    """
    Fit repeated logistic-regression models for one N_nearby selection.

    Parameters
    ----------
    restrict_input_to_selection : bool
        If True, fit only sources satisfying N_nearby <= threshold.
        If False, fit all visible sources and predict membership in the
        N_nearby-selected candidate subset.
    """
    if restrict_input_to_selection:
        selection_mask = visible_mask & data["uncertainty_count"].le(n_nearby)
    else:
        selection_mask = visible_mask.copy()

    target_base_mask = visible_mask & data["uncertainty_count"].le(n_nearby)

    X = build_features(data.loc[selection_mask], feature_set).reset_index(drop=True)

    y_by_rmag = []
    candidate_counts = []

    for rmag_column in rmag_columns:
        target_mask = target_base_mask & data[rmag_column].le(RMAG_LIMIT)
        y = target_mask.loc[selection_mask].astype(int).reset_index(drop=True)

        y_by_rmag.append(y)
        candidate_counts.append(int(target_mask.sum()))

    candidate_counts = np.asarray(candidate_counts)

    print("\n" + "=" * 78)
    print(f"Feature set: {feature_set}")
    print(f"N_nearby ≤ {n_nearby}")
    print(f"Input sample restricted to selection: {restrict_input_to_selection}")
    print(f"Input sources: {len(X):,}")
    print(f"Mean target count: {candidate_counts.mean():.0f} ± {candidate_counts.std():.0f}")
    print(f"Target-count range: {candidate_counts.min():.0f}–{candidate_counts.max():.0f}")

    model = make_pipeline(
        PolynomialFeatures(degree=DEGREE, include_bias=False),
        LogisticRegression(
            C=C_VALUE,
            solver="lbfgs",
            max_iter=10_000,
            random_state=RANDOM_STATE,
        ),
    )

    validation_y, validation_scores, metrics, curve_summary = repeated_validation(
        model,
        X,
        y_by_rmag,
    )

    threshold_index = np.argmin(np.abs(THRESHOLDS - THRESHOLD))

    print(f"\n===== Training Predictions (threshold={THRESHOLD:.1f}) =====")
    print(
        "Mean training target count: "
        f"{metrics['n_positive_train'].mean():.0f} ± "
        f"{metrics['n_positive_train'].std():.0f}"
    )
    print(
        "Training N_selected: "
        f"{curve_summary['train_npred_mean'][threshold_index]:.0f} ± "
        f"{curve_summary['train_npred_std'][threshold_index]:.0f}"
    )
    print(
        "Training precision: "
        f"{curve_summary['train_precision_mean'][threshold_index]:.2f} ± "
        f"{curve_summary['train_precision_std'][threshold_index]:.2f}"
    )
    print(
        "Training recall: "
        f"{curve_summary['train_recall_mean'][threshold_index]:.2f} ± "
        f"{curve_summary['train_recall_std'][threshold_index]:.2f}"
    )

    print("\n===== Repeated Held-Out Validation =====")
    print(f"Successful fits: {curve_summary['n_successful_fits']:,}")
    print(
        "Mean validation target count: "
        f"{metrics['n_positive_test'].mean():.0f} ± "
        f"{metrics['n_positive_test'].std():.0f}"
    )
    print(
        "Validation precision: "
        f"{curve_summary['test_precision_mean'][threshold_index]:.2f} ± "
        f"{curve_summary['test_precision_std'][threshold_index]:.2f}"
    )
    print(
        "Validation recall: "
        f"{curve_summary['test_recall_mean'][threshold_index]:.2f} ± "
        f"{curve_summary['test_recall_std'][threshold_index]:.2f}"
    )
    print(
        "Validation ROC-AUC: "
        f"{np.nanmean(metrics['roc_auc']):.2f} ± {np.nanstd(metrics['roc_auc']):.2f}"
    )
    print(
        "Validation average precision: "
        f"{np.nanmean(metrics['average_precision']):.2f} ± "
        f"{np.nanstd(metrics['average_precision']):.2f}"
    )

    print(f"\n===== All-Data Predictions (threshold={THRESHOLD:.1f}) =====")
    print(
        "Mean target count: "
        f"{candidate_counts.mean():.0f} ± {candidate_counts.std():.0f}"
    )
    print(
        "All-data N_selected: "
        f"{curve_summary['all_npred_mean'][threshold_index]:.0f} ± "
        f"{curve_summary['all_npred_std'][threshold_index]:.0f}"
    )
    print(
        "All-data precision: "
        f"{curve_summary['all_precision_mean'][threshold_index]:.2f} ± "
        f"{curve_summary['all_precision_std'][threshold_index]:.2f}"
    )
    print(
        "All-data recall: "
        f"{curve_summary['all_recall_mean'][threshold_index]:.2f} ± "
        f"{curve_summary['all_recall_std'][threshold_index]:.2f}"
    )

    X_final = pd.concat([X] * N_RMAG_REALIZATIONS, ignore_index=True)
    y_final = pd.concat(y_by_rmag, ignore_index=True)

    model.fit(X_final, y_final)
    print_logistic_equation(model, X.columns, n_nearby, feature_set)

    return {
        "model": model,
        "X": X,
        "candidate_counts": candidate_counts,
        "validation_y": validation_y,
        "validation_scores": validation_scores,
        "metrics": metrics,
        "curve_summary": curve_summary,
    }


# =============================================================================
# Machine-learning analysis: mass and distance only
# =============================================================================
fig, axes = plt.subplots(
    nrows=2,
    ncols=len(N_NEARBY_VALUES),
    figsize=(5.0 * len(N_NEARBY_VALUES), 9.5),
    squeeze=False,
)

ml_results_mass_distance = {}

for column, n_nearby in enumerate(N_NEARBY_VALUES):
    result = run_ml_case(
        lisa,
        n_nearby=n_nearby,
        feature_set="mass_distance",
        restrict_input_to_selection=True,
    )

    ml_results_mass_distance[n_nearby] = result

    plot_ml_case(
        n_nearby,
        axes[:, column],
        result["curve_summary"],
        result["candidate_counts"].mean(),
        result["candidate_counts"].std(),
    )

fig.tight_layout()
save_figure(fig, "recall_precision_mass_distance.pdf")
plt.show()
    

# =============================================================================
# Machine-learning analysis: all logarithmic physical parameters
# =============================================================================
"""fig, axes = plt.subplots(
    nrows=2,
    ncols=len(N_NEARBY_VALUES),
    figsize=(5.0 * len(N_NEARBY_VALUES), 9.5),
    squeeze=False,
)

ml_results_all_parameters = {}

for column, n_nearby in enumerate(N_NEARBY_VALUES):
    result = run_ml_case(
        lisa,
        n_nearby=n_nearby,
        feature_set="log_all_parameters",
        restrict_input_to_selection=False,
    )

    ml_results_all_parameters[n_nearby] = result

    plot_ml_case(
        n_nearby,
        axes[:, column],
        result["curve_summary"],
        result["candidate_counts"].mean(),
        result["candidate_counts"].std(),
    )

fig.tight_layout()
save_figure(fig, "recall_precision_all_parameters.pdf")
plt.show()"""


# ============================================================================
# PROPER-MOTION MODEL AND CANDIDATE FRACTION ANALYSIS
# ============================================================================

# Galactocentric reference frame used consistently for positions and velocities.
GC_FRAME = Galactocentric(
    galcen_distance=8.122 * u.kpc,
    galcen_v_sun=CartesianDifferential(
        [11.1, 229.0 + 12.24, 7.25] * u.km / u.s
    ),
    z_sun=20.8 * u.pc,
)

# Mean rotational velocities and isotropic 1D velocity dispersions in km/s.
V_ROT = {"thin": 223.0, "thick": 183.0, "halo": 0.0, "bulge": 0.0}
SIGMA_V = {"thin": 30.0, "thick": 50.0, "halo": 100.0, "bulge": 120.0}

# Galactic-component boundaries in kpc.
R_BULGE = 1.0
Z_THIN = 0.3
Z_THICK = 1.0

N_PM_DRAWS = 1000


def proper_motion_components(l_deg, b_deg, distance_kpc, rng=None):
    """Return pm_l_cosb, pm_b, and Galactic component for source arrays."""
    galactic_coord = SkyCoord(
        l=l_deg * u.deg,
        b=b_deg * u.deg,
        distance=distance_kpc * u.kpc,
        frame=Galactic(),
    )
    galactocentric_coord = galactic_coord.transform_to(GC_FRAME)

    x = galactocentric_coord.x.to_value(u.kpc)
    y = galactocentric_coord.y.to_value(u.kpc)
    z = galactocentric_coord.z.to_value(u.kpc)

    radius_3d = np.sqrt(x**2 + y**2 + z**2)
    radius_xy = np.hypot(x, y)
    abs_z = np.abs(z)

    component = np.where(
        radius_3d < R_BULGE,
        "bulge",
        np.where(
            abs_z < Z_THIN,
            "thin",
            np.where(abs_z < Z_THICK, "thick", "halo"),
        ),
    )

    v_rotation = np.array([V_ROT[label] for label in component], dtype=float)
    safe_radius_xy = np.where(radius_xy > 0.0, radius_xy, np.finfo(float).eps)

    # Circular streaming motion in Galactocentric Cartesian coordinates.
    vx = v_rotation * y / safe_radius_xy
    vy = -v_rotation * x / safe_radius_xy
    vz = np.zeros_like(vx)

    if rng is not None:
        sigma_velocity = np.array(
            [SIGMA_V[label] for label in component],
            dtype=float,
        )
        vx += sigma_velocity * rng.standard_normal(vx.shape)
        vy += sigma_velocity * rng.standard_normal(vy.shape)
        vz += sigma_velocity * rng.standard_normal(vz.shape)

    star_coord = SkyCoord(
        x=x * u.kpc,
        y=y * u.kpc,
        z=z * u.kpc,
        v_x=vx * u.km / u.s,
        v_y=vy * u.km / u.s,
        v_z=vz * u.km / u.s,
        frame=GC_FRAME,
        representation_type="cartesian",
        differential_type="cartesian",
    ).transform_to(Galactic())

    return (
        star_coord.pm_l_cosb.to_value(u.mas / u.yr),
        star_coord.pm_b.to_value(u.mas / u.yr),
        component,
    )


# ----------------------------------------------------------------------------
# Monte Carlo proper-motion realizations from the Galactic velocity model.
# ----------------------------------------------------------------------------

n_sources = len(lisa)
pm_l_draws = np.empty((N_PM_DRAWS, n_sources))
pm_b_draws = np.empty((N_PM_DRAWS, n_sources))

galactic_longitude = lisa["gall"].to_numpy()
galactic_latitude = lisa["galb"].to_numpy()
distance_kpc = lisa["d"].to_numpy()

for draw_index in range(N_PM_DRAWS):
    rng = np.random.default_rng(draw_index)

    pm_l_draws[draw_index], pm_b_draws[draw_index], component = (
        proper_motion_components(
            galactic_longitude,
            galactic_latitude,
            distance_kpc,
            rng=rng,
        )
    )

pm_draws = np.hypot(pm_l_draws, pm_b_draws)

lisa["pm"] = pm_draws.mean(axis=0)
lisa["pm_sigma_kinematic"] = pm_draws.std(axis=0)
lisa["component"] = component

# Combine intrinsic kinematic scatter with the astrometric uncertainty if present.
if "pm_error" in lisa.columns:
    lisa["pm_total_uncertainty"] = np.hypot(
        lisa["pm_error"],
        lisa["pm_sigma_kinematic"],
    )
else:
    lisa["pm_total_uncertainty"] = lisa["pm_sigma_kinematic"]


# ----------------------------------------------------------------------------
# Fractions of visible candidates with mu >= PM_LIMIT across r-band realizations.
#
# visible_mask must correspond row-by-row to lisa. It is normally defined earlier
# in the script from the HEALPix map, e.g.:
#
# visible_mask = pixel_counts[pix_all] > 0
# ----------------------------------------------------------------------------

candidate_groups = [
    (
        r"$N_{\mathrm{nearby}} \leq 10$",
        lisa["uncertainty_count"] <= 10,
    ),
    (
        r"$10 < N_{\mathrm{nearby}} \leq 100$",
        (lisa["uncertainty_count"] > 10)
        & (lisa["uncertainty_count"] <= 100),
    ),
]

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

for ax, (label, nearby_mask) in zip(axes, candidate_groups):
    candidate_percentages = []
    candidate_mask = nearby_mask & visible_mask

    for realization in range(N_RMAG_REALIZATIONS):
        magnitude_mask = candidate_mask & (
            lisa[f"rmag_{realization}"] <= RMAG_LIMIT
        )

        n_selected = magnitude_mask.sum()
        n_pm_selected = (lisa.loc[magnitude_mask, "pm"] >= PM_LIMIT).sum()

        candidate_percentages.append(
            100.0 * n_pm_selected / n_selected if n_selected > 0 else np.nan
        )

    mean_fraction = np.nanmean(candidate_percentages)
    std_fraction = np.nanstd(candidate_percentages)

    ax.hist(
        candidate_percentages,
        bins=10,
        color="steelblue",
        alpha=0.75,
        edgecolor="white",
    )
    ax.set_title(label, fontsize=14)
    ax.set_xlabel(
        rf"Candidates with $\mu \geq {PM_LIMIT:.0f}$ mas yr$^{{-1}}$ (%)",
        fontsize=12,
    )
    ax.set_ylabel("Number of realizations", fontsize=12)
    ax.text(
        0.05,
        0.95,
        rf"$\bar{{P}} = {mean_fraction:.1f} \pm {std_fraction:.1f}\%$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        bbox={"facecolor": "none", "edgecolor": "none"},
    )

fig.tight_layout()
fig.savefig("candidate_pm.pdf", dpi=FIG_DPI, bbox_inches="tight")
plt.show()
