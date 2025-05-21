# -----------------
# Tunable constants
# -----------------
FARE_CONVERGENCE_TOLERANCE = 10.0  # initial fare optimisation is converged once all mean fares shift by less than this amount (USD)
MAX_RANGE_PAYLOAD_PROPORTION = 0.8  # proportion of max payload used for calculating representative ranges (used for wide-body a/c only, fixed to 1.0 for narrow-body a/c)
MIN_PAX_LOAD_FACTOR = 0.8  # minimum load factor (expected tickets sold / available seats) for a route to be considered viable
LOITER_T_SEC = 1800  # 30 minutes in seconds
DIVERSION_DIST_METRES = 185200  # 100 nautical miles in metres
PASSENGER_MASS_KG = 100  # average passenger mass in kg (including baggage)
CURFEW_HOURS = 7  # assume airports are closed between 11pm and 6am
FUEL_DENSITY_KG_LITRE = 0.8  # kg/l
FUEL_ENERGY_MJ_KG = 43.1  # MJ/kg
PASSENGER_AC_FUEL_PROPORTION = 0.77  # proportion of fuel used for passenger a/c (vs cargo and private a/c) - neglect military a/c
MIN_INIT_PLANES_PER_AL = 10  # min initial number of planes assigned to an airline (unless country has fewer planes than this)
MAX_EXPANSION_PLANES = 1  # max number of planes an airline can add in a single year
MIN_OCCUPANCY_FOR_EXPANSION = 0.7  # minimum occupancy on an itinerary to add another a/c
MAX_EXPANSION_PROPORTION = 0.10  # max proportion of fleet size that an airline can add in a single year
PRICE_ELAS_LH_THRESHOLD = 4828032  # 3000 miles in metres
INCOME_ELAS_THRESHOLDS = {
    "short": 0.0,
    "medium": 1609344.0,
    "long": 4828032.0,
    "ultra_long": 8046720.0
}  # 0 / 1000 / 3000 / 5000 miles in metres
GNI_THRESHOLDS = {
    "lower_middle": 1166,
    "upper_middle": 4526,
    "high": 14005
}  # World Bank GNI per capita thresholds for country income levels (USD)
# AC_SIZE_EQUIVALENCE = {
#     "size_0": [3,2,1],
#     "size_1": [3,2,3],
#     "size_2": [6,5,3],
#     "size_4": [4,3,5],
#     "size_5": [4,3,6],
# }  # [number of size_a a/c to replace, number of larger a/c to add, size class of larger a/c]
ROUTE_MAX_SINGLE_SZ = {
    "size_0": 3,
    "size_1": 3,
    "size_2": -1,
    "size_3": -1,
    "size_4": 6,
    "size_5": -1,
    "size_6": -1,
}  # use to force airlines to wait for demand to add larger a/c, rather than repeatedly adding smaller a/c (-1 => no limit)

# ------------------
# Physical constants
# ------------------
US_GALLONS_PER_L = 0.264172  # US gallons in one litre


# --------------------
# Calculated constants
# --------------------
FUEL_GALLONS_PER_KG = US_GALLONS_PER_L /FUEL_DENSITY_KG_LITRE  # 1kg of fuel in gallons ~= 0.33
OP_HRS_PER_YEAR = (24-CURFEW_HOURS) * 365  # airport operational hours per year
