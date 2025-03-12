import numpy as np
import pandas as pd
from pygeodesy import sphericalNvector as snv


def initialise_routes(
    city_data: pd.DataFrame,
    city_pair_data: pd.DataFrame,
    price_elasticities: pd.DataFrame,
    income_elasticities: pd.DataFrame,
    population_elasticity: float,
) -> pd.DataFrame:
    """
    Initialise the routes between cities with demand and elasticity data.

    Parameters
    ----------
    city_data : pd.DataFrame
    city_pair_data : pd.DataFrame
    price_elasticities : pd.DataFrame
    income_elasticities : pd.DataFrame
    population_elasticity : float

    Returns
    -------
    city_pair_data : pd.DataFrame
    """
    # arbitrary thresholds for flight types in meters
    price_elas_LH_threshold = 3000 * 1609.344
    income_elas_thresholds = {
        "short": 0.0,
        "medium": 1000*1609.344,
        "long": 3000*1609.344,
        "ultra_long": 5000*1609.344
    }

    # World Bank GNI thresholds for country income levels
    GNI_thresholds = {
        "lower_middle": 1166,
        "upper_middle": 4526,
        "high": 14005
    }

    # initialise whole columns
    city_pair_data["Mean_Fare_USD"] = city_pair_data["Fare_Est"]
    city_pair_data["Total_Demand"] = city_pair_data["BaseYearODDemandPax_Est"]

    # initialise lists using 32-bit data types where possible for memory efficiency
    n = len(city_pair_data)
    remove_idx = []
    distance = np.zeros(n, dtype=np.float32)
    origin_long = np.zeros(n, dtype=np.float32)
    origin_lat = np.zeros(n, dtype=np.float32)
    destination_long = np.zeros(n, dtype=np.float32)
    destination_lat = np.zeros(n, dtype=np.float32)
    price_elasticity_route = np.zeros(n, dtype=np.float32)
    price_elasticity_national = np.zeros(n, dtype=np.float32)
    origin_income_elasticity = np.zeros(n, dtype=np.float32)
    destination_income_elasticity = np.zeros(n, dtype=np.float32)
    seat_flights_per_year = np.zeros(n, dtype=np.int32)
    static_demand_factor = np.zeros(n, dtype=np.float64)
    exp_utility_sum = np.zeros(n, dtype=np.float64)
    international = np.zeros(n, dtype=bool)

    for idx, route in city_pair_data.iterrows():
        origin_id = route["OriginCityID"]
        destination_id = route["DestinationCityID"]

        if (
            (origin_id not in city_data.index)
            or (destination_id not in city_data.index)
            or (origin_id == destination_id)
            or (origin_id == 0)
            or (destination_id == 0)
        ):
            # flag for removal
            remove_idx.append(idx)
        else:
            origin = city_data.loc[origin_id]
            destination = city_data.loc[destination_id]

            # check if the route is international
            international[idx] = origin["Country"] != destination["Country"]

            # calculate great circle distance between origin and destination cities
            distance[idx] = calc_great_circle_distance(
                origin["Latitude"],
                origin["Longitude"],
                destination["Latitude"],
                destination["Longitude"]
            )

            # store city coordinates to make plotting easier
            origin_lat[idx] = origin["Latitude"]
            origin_long[idx] = origin["Longitude"]
            destination_lat[idx] = destination["Latitude"]
            destination_long[idx] = destination["Longitude"]

            # retrieve the relevant price_elasticity of demand - note OD_1 <= OD_2
            price_elasticities_idx = price_elasticities.loc[
                (price_elasticities["OD_1"] == min(origin["Region"], destination["Region"]))
                & (price_elasticities["OD_2"] == max(origin["Region"], destination["Region"]))
            ].index[0]
            if (distance[idx] < price_elas_LH_threshold):
                haul = "SH"
            else:
                haul = "LH"
            price_elasticity_route[idx] = price_elasticities.loc[price_elasticities_idx, f"route_{haul}"]
            price_elasticity_national[idx] = price_elasticities.loc[price_elasticities_idx, f"national_{haul}"]

            # retrieve the relevant income_elasticity of demand
            origin_income_elasticity[idx], destination_income_elasticity[idx] = route_income_elasticity(
                income_elasticities,
                distance[idx],
                income_elas_thresholds,
                GNI_thresholds,
                origin["Country"],
                destination["Country"],
                origin["Income_USDpercap"],
                destination["Income_USDpercap"],
            )

            # calculate base year static demand factor
            static_demand_factor[idx] = calc_static_demand_factor(
                origin["Population"],
                destination["Population"],
                origin["BaseYearPopulation"],
                destination["BaseYearPopulation"],
                origin["Income_USDpercap"],
                destination["Income_USDpercap"],
                origin["BaseYearIncome"],
                destination["BaseYearIncome"],
                origin_income_elasticity[idx],
                destination_income_elasticity[idx],
                population_elasticity,
            )
    
    city_pair_data['Great_Circle_Distance_m'] = distance.astype('float32')
    city_pair_data['Origin_Latitude'] = origin_lat.astype('float32')
    city_pair_data['Origin_Longitude'] = origin_long.astype('float32')
    city_pair_data['Destination_Latitude'] = destination_lat.astype('float32')
    city_pair_data['Destination_Longitude'] = destination_long.astype('float32')
    city_pair_data['Price_Elasticity_Route'] = price_elasticity_route.astype('float32')
    city_pair_data['Price_Elasticity_National'] = price_elasticity_national.astype('float32')
    city_pair_data['Origin_Income_Elasticity'] = origin_income_elasticity.astype('float32')
    city_pair_data['Destination_Income_Elasticity'] = destination_income_elasticity.astype('float32')
    city_pair_data["Seat_Flights_perYear"] = seat_flights_per_year.astype('int32')
    city_pair_data["Static_Demand_Factor"] = static_demand_factor
    city_pair_data["Exp_Utility_Sum"] = exp_utility_sum.astype('float64')
    city_pair_data["International"] = international.astype('bool')

    # remove flagged routes
    city_pair_data.drop(remove_idx, inplace=True)

    return city_pair_data


def calc_great_circle_distance(origin_lat, origin_long, destination_lat, destination_long) -> float:
    """
    Calculate the great circle distance at the earth's surface in meters between the origin and destination cities.

    Parameters
    ----------
    origin_lat : float
        Latitude of the origin city
    origin_long : float
        Longitude of the origin city
    destination_lat : float
        Latitude of the destination city
    destination_long : float
        Longitude of the destination city
    
    Returns
    -------
    distance : float
        Great circle distance between the origin and destination cities in meters
    """
    # TODO: generate waypoints between origin and destination to route around airspace restrictions
    origin_coords = snv.LatLon(origin_lat, origin_long)
    destination_coords = snv.LatLon(destination_lat, destination_long)
    return origin_coords.distanceTo(destination_coords)


def route_income_elasticity(
        income_elasticities: pd.DataFrame,
        distance: float,
        income_elas_thresholds: dict,
        GNI_thresholds: dict,
        origin_country_code: int,
        destination_country_code: int,
        origin_income_USDpercap: float,
        destination_income_USDpercap: float,
    ) -> tuple[float, float]:
    """
    Retrieve the appropriate income elasticity values to the origin and destination cities.

    Parameters
    ----------
    income_elasticities : pd.DataFrame
    distance : float
    income_elas_thresholds : dict
    GNI_thresholds : dict
    country_code : int
    origin_income_USDpercap : float
    destination_income_USDpercap : float

    Returns
    -------
    origin_income_elasticity : float
    destination_income_elasticity : float
    """
    if distance < income_elas_thresholds["medium"]:
        haul = "SH"
    elif distance < income_elas_thresholds["long"]:
        haul = "MH"
    elif distance < income_elas_thresholds["ultra_long"]:
        haul = "LH"
    else:
        haul = "ULH"
    
    origin_income_elasticity = city_income_elasticity(
        income_elasticities,
        haul,
        GNI_thresholds,
        origin_country_code,
        origin_income_USDpercap,
    )
    destination_income_elasticity = city_income_elasticity(
        income_elasticities,
        haul,
        GNI_thresholds,
        destination_country_code,
        destination_income_USDpercap,
    )

    return origin_income_elasticity, destination_income_elasticity


def city_income_elasticity(
    income_elasticities: pd.DataFrame,
    haul: str,
    GNI_thresholds: dict,
    country_code: int,
    income_USDpercap: float,
) -> float:
    """
    Determine the income elasticity of demand for a city based on its income level and length of route it is associated with.
    Used in route_income_elasticity to avoid code repetition.

    Parameters
    ----------
    income_elasticities : pd.DataFrame
    haul : str
    GNI_thresholds : dict
    country_code : int
    income_USDpercap : float

    Returns
    -------
    income_elasticity : float
    """
    # TODO: determine U.S. country code programmatically

    if country_code == 204:  # U.S. country code
        income_elasticity = income_elasticities.loc[
            income_elasticities["CountryType"] == "US", f"elasticity_{haul}"
        ].iloc[0]
    elif income_USDpercap < GNI_thresholds["lower_middle"]:
        income_elasticity = income_elasticities.loc[
            income_elasticities["CountryType"] == "Developing", f"elasticity_{haul}"
        ].iloc[0]
    elif income_USDpercap < GNI_thresholds["high"]:
        # linearly interpolate between developing and developed country elasticities
        gradient = (
            income_elasticities.loc[
                income_elasticities["CountryType"] == "Developed",
                f"elasticity_{haul}",
            ].iloc[0]
            - income_elasticities.loc[
                income_elasticities["CountryType"] == "Developing",
                f"elasticity_{haul}",
            ].iloc[0]
        ) / (GNI_thresholds["high"] - GNI_thresholds["lower_middle"])
        income_elasticity = income_elasticities.loc[
            income_elasticities["CountryType"] == "Developing", f"elasticity_{haul}"
        ].iloc[0] + gradient * (income_USDpercap - GNI_thresholds["lower_middle"])
    else:
        income_elasticity = income_elasticities.loc[
            income_elasticities["CountryType"] == "Developed", f"elasticity_{haul}"
        ].iloc[0]

    return income_elasticity


def calc_static_demand_factor(
    origin_population: float,
    destination_population: float,
    origin_base_population: float,
    destination_base_population: float,
    origin_income_USDpercap: float,
    destination_income_USDpercap: float,
    origin_base_income_USDpercap: float,
    destination_base_income_USDpercap: float,
    origin_income_elasticity: float,
    destination_income_elasticity: float,
    population_elasticity: float
) -> float:
    """
    Calculate the total route demand factor from effects independent of fare.

    Parameters
    ----------
    origin_population : float
    destination_population : float
    origin_base_population : float
    destination_base_population : float
    origin_income_USDpercap : float
    destination_income_USDpercap : float
    origin_base_income_USDpercap : float
    destination_base_income_USDpercap : float
    origin_income_elasticity : float
    destination_income_elasticity : float
    population_elasticity : float

    Returns
    -------
    static_demand_factor : float
    """
    # assume demand originating from each city is proportional to the city's population * income
    origin_demand_weight = (
        (origin_population * origin_income_USDpercap)
        / ((origin_population * origin_income_USDpercap)
            + (destination_population * destination_income_USDpercap))
    )
    destination_demand_weight = 1 - origin_demand_weight
    
    income_factor_origin = 1 + (
        ((origin_income_USDpercap - origin_base_income_USDpercap) 
            / origin_base_income_USDpercap) * origin_income_elasticity)

    income_factor_destination = 1 + (
        ((destination_income_USDpercap - destination_base_income_USDpercap) 
            / destination_base_income_USDpercap) * destination_income_elasticity)

    # total income factor weighted by population * income
    income_factor = (
        (origin_demand_weight * income_factor_origin)
        + (destination_demand_weight * income_factor_destination)
    )

    population_factor_origin = 1 + (
        ((origin_population - origin_base_population) 
            / origin_base_population) * population_elasticity)

    population_factor_destination = 1 + (
        ((destination_population - destination_base_population) 
            / destination_base_population) * population_elasticity)

    # total population factor weighted by population * income
    population_factor = (
        (origin_demand_weight * population_factor_origin)
        + (destination_demand_weight * population_factor_destination)
    )

    return income_factor * population_factor


def choose_fuel_stop(
    city_data: pd.DataFrame,
    origin: pd.Series,
    destination: pd.Series,
    aircraft_range: float,
    min_runway_m: float
) -> int:
    """
    Choose a fuel stop city for a route based on range and runway length requirements.

    Parameters
    ----------
    city_data : pd.DataFrame
    origin : pd.Series
    destination : pd.Series
    aircraft_range : float
    min_runway_m : float

    Returns
    -------
    fuel_stop : int
        City ID of the chosen fuel stop or -1 if no suitable city is found
    """
    # calculate the midpoint between the origin and destination
    origin_coords = snv.LatLon(
        origin["Latitude"],
        origin["Longitude"]
    )
    destination_coords = snv.LatLon(
        destination["Latitude"],
        destination["Longitude"]
    )

    midpoint = origin_coords.midpointTo(destination_coords)

    # find the nearest city to the midpoint that has a long enough runway
    fuel_stop = -1
    min_distance = np.inf
    for city_id, city in city_data.iterrows():
        city_coords = snv.LatLon(
            city["Latitude"],
            city["Longitude"],
        )
        distance = midpoint.distanceTo(city_coords)
        if distance < min_distance:
            # check that runway and leg distances are suitable for the aircraft
            if (
                city["LongestRunway_m"] > min_runway_m
                and origin_coords.distanceTo(city_coords) < aircraft_range
                and city_coords.distanceTo(destination_coords) < aircraft_range
            ):
                fuel_stop = city_id
                min_distance = distance
    return fuel_stop
