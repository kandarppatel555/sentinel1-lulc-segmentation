"""
Shared configuration for the Sentinel-1 LULC project.

Everything that must be IDENTICAL between dataset building, training and the
prototype app lives here (class scheme, compositing rules, normalisation).
Changing something here means rebuilding the dataset and retraining.
"""

# ---------------------------------------------------------------------------
# Earth Engine collections
# ---------------------------------------------------------------------------
S1_COLLECTION = "COPERNICUS/S1_GRD"          # calibrated, ortho-corrected, values in dB
WORLDCOVER_ASSET = "ESA/WorldCover/v200"      # ESA WorldCover 2021, 10 m
LABEL_YEAR = 2021                             # WorldCover v200 reference year -> S1 year for training

# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------
PIXEL_SIZE_M = 10          # native resolution of both S1 GRD (IW) and WorldCover
REGION_SIZE_PX = 768       # training region = 7.68 km x 7.68 km
PATCH_SIZE_PX = 256        # training crop size (2.56 km)
APP_SIZE_PX = 512          # prototype area = 5.12 km x 5.12 km (must be divisible by 32)

# ---------------------------------------------------------------------------
# Sentinel-1 compositing
# ---------------------------------------------------------------------------
S1_BANDS = ["VV", "VH"]                  # stored on disk
INPUT_CHANNELS = ["VV", "VH", "VV_minus_VH"]  # fed to the model (ratio computed on the fly)
INSTRUMENT_MODE = "IW"
# Mask the noisy near/far range edges of each IW swath (degrees of incidence angle)
ANGLE_MIN = 30.6
ANGLE_MAX = 45.0
NODATA_VALUE = -9999.0     # value used to unmask on the Earth Engine side

# Clip ranges in dB applied before normalisation (removes extreme outliers,
# e.g. corner reflectors in cities or speckle spikes)
DB_CLIP = {"VV": (-30.0, 5.0), "VH": (-37.0, -5.0), "VV_minus_VH": (-5.0, 20.0)}

# ---------------------------------------------------------------------------
# Class scheme
# WorldCover's 11 classes merged into 6 that radar can plausibly separate.
# ---------------------------------------------------------------------------
IGNORE_INDEX = 255

CLASS_NAMES = [
    "Tree cover",          # 0
    "Shrub & grassland",   # 1
    "Cropland",            # 2
    "Built-up",            # 3
    "Bare / sparse",       # 4
    "Water",               # 5
]
NUM_CLASSES = len(CLASS_NAMES)

# Hex colours (WorldCover-like palette so maps are easy to read)
CLASS_COLORS = [
    "#006400",  # Tree cover
    "#FFBB22",  # Shrub & grassland
    "#F096FF",  # Cropland
    "#FA0000",  # Built-up
    "#B4B4B4",  # Bare / sparse
    "#0064C8",  # Water
]

# WorldCover code -> merged class index
WORLDCOVER_TO_CLASS = {
    10: 0,    # Tree cover               -> Tree cover
    20: 1,    # Shrubland                -> Shrub & grassland
    30: 1,    # Grassland                -> Shrub & grassland
    40: 2,    # Cropland                 -> Cropland
    50: 3,    # Built-up                 -> Built-up
    60: 4,    # Bare / sparse vegetation -> Bare / sparse
    70: IGNORE_INDEX,  # Snow and ice    -> ignored (seasonal, radar-ambiguous)
    80: 5,    # Permanent water bodies   -> Water
    90: 1,    # Herbaceous wetland       -> Shrub & grassland
    95: 0,    # Mangroves                -> Tree cover
    100: 1,   # Moss and lichen          -> Shrub & grassland
}

# ---------------------------------------------------------------------------
# Training regions (name, lat, lon, split).
# Splitting by REGION (not by random patch) prevents spatial leakage.
# Mannheim is held out in the test split so the app demo runs on unseen data.
# ---------------------------------------------------------------------------
REGIONS = [
    # --- Cities (built-up) ---
    ("berlin",            52.520,  13.405, "train"),
    ("hamburg",           53.550,  10.000, "train"),
    ("munich",            48.137,  11.575, "train"),
    ("cologne",           50.940,   6.960, "val"),
    ("frankfurt",         50.110,   8.680, "train"),
    ("stuttgart",         48.780,   9.180, "train"),
    ("mannheim",          49.490,   8.470, "test"),
    ("leipzig",           51.340,  12.370, "train"),
    ("paris",             48.857,   2.352, "train"),
    ("lyon",              45.760,   4.835, "test"),
    ("amsterdam",         52.370,   4.900, "train"),
    ("brussels",          50.850,   4.350, "train"),
    ("vienna",            48.210,  16.370, "train"),
    ("prague",            50.075,  14.440, "train"),
    ("warsaw",            52.230,  21.010, "val"),
    ("milan",             45.464,   9.190, "train"),
    ("madrid",            40.417,  -3.700, "train"),
    ("copenhagen",        55.676,  12.568, "train"),
    ("london",            51.507,  -0.128, "train"),
    ("budapest",          47.498,  19.040, "train"),
    # --- Cropland ---
    ("magdeburg_boerde",  52.050,  11.350, "train"),
    ("beauce",            48.200,   1.600, "train"),
    ("po_valley",         45.130,  10.000, "train"),
    ("hungarian_plain",   47.000,  20.500, "val"),
    ("poland_agri",       51.800,  17.500, "train"),
    ("lower_saxony",      52.600,   9.000, "train"),
    ("picardy",           49.800,   2.800, "test"),
    ("castilla",          41.600,  -4.700, "train"),
    ("jutland",           56.200,   9.300, "train"),
    ("baragan",           44.500,  27.300, "train"),
    # --- Forest ---
    ("black_forest",      48.200,   8.200, "train"),
    ("bavarian_forest",   49.000,  13.400, "train"),
    ("harz",              51.750,  10.600, "val"),
    ("landes",            44.200,  -0.900, "train"),
    ("bialowieza",        52.700,  23.850, "train"),
    ("palatinate_forest", 49.300,   7.850, "train"),
    ("thuringian_forest", 50.650,  10.800, "test"),
    # --- Water / wetland / coast ---
    ("lake_constance",    47.600,   9.400, "train"),
    ("ijsselmeer",        52.700,   5.300, "train"),
    ("jadebusen",         53.450,   8.200, "train"),
    ("lake_balaton",      46.850,  17.700, "test"),
    ("mueritz",           53.450,  12.700, "val"),
    ("danube_delta",      45.150,  29.300, "train"),
    ("lake_geneva",       46.450,   6.550, "train"),
    # --- Bare / sparse ---
    ("bardenas_reales",   42.200,  -1.500, "val"),
    ("tabernas",          37.050,  -2.400, "train"),
    ("garzweiler_mine",   51.060,   6.500, "train"),
    ("lusatia_mines",     51.600,  14.400, "test"),
    # --- Grassland / shrub / heath ---
    ("scottish_highlands",57.000,  -4.500, "train"),
    ("dartmoor",          50.570,  -3.920, "val"),
    ("lueneburg_heath",   53.170,   9.950, "train"),
    ("extremadura",       39.300,  -6.200, "train"),
    ("massif_central",    45.300,   2.900, "train"),
]

MONTHS = list(range(1, 13))
