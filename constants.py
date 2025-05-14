# -----------------
# Tunable constants
# -----------------
FARE_CONVERGENCE_TOLERANCE = 10.0  # initial fare optimisation is converged once all mean fares shift by less than this amount (USD)
MAX_RANGE_PAYLOAD_PROPORTION = 0.8  # proportion of max payload used for calculating representative ranges
MIN_PAX_LOAD_FACTOR = 0.8  # minimum load factor (expected tickets sold / available seats) for a route to be considered viable
LOITER_T_SEC = 1800  # 30 minutes in seconds
DIVERSION_DIST_METRES = 185200  # 100 nautical miles in metres
PASSENGER_MASS_KG = 100  # average passenger mass in kg (including baggage)
CURFEW_HOURS = 7  # assume airports are closed between 11pm and 6am
FUEL_DENSITY_KG_LITRE = 0.8  # kg/l
FUEL_ENERGY_MJ_KG = 43.1  # MJ/kg
MIN_INIT_PLANES_PER_AL = 10  # min initial number of planes assigned to an airline (unless country has fewer planes than this)
MAX_EXPANSION_PLANES = 20  # max number of planes an airline can add in a single year
MAX_EXPANSION_PROPORTION = 0.2  # max proportion of fleet size that an airline can add in a single year
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

# ------------------
# Physical constants
# ------------------
US_GALLONS_PER_L = 0.264172  # US gallons in one litre


# --------------------
# Calculated constants
# --------------------
FUEL_GALLONS_PER_KG = US_GALLONS_PER_L /FUEL_DENSITY_KG_LITRE  # 1kg of fuel in gallons ~= 0.33
OP_HRS_PER_YEAR = (24-CURFEW_HOURS) * 365  # airport operational hours per year
