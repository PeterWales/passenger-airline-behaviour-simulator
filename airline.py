import pandas as pd
import numpy as np
import route
import aircraft as acft
import demand
import reassignment
from skopt import gp_minimize
import datetime
import math
from constants import (
    MIN_INIT_PLANES_PER_AL,
    MIN_PAX_LOAD_FACTOR,
    OP_HRS_PER_YEAR,
    MAX_EXPANSION_PLANES,
    MAX_EXPANSION_PROPORTION,
)
import warnings

warnings.filterwarnings("ignore", message="The objective has been evaluated at point*", category=UserWarning)


def initialise_airlines(
    fleet_data: pd.DataFrame,
    country_data: pd.DataFrame,
    run_parameters: pd.DataFrame,
    regions: list | None,
) -> pd.DataFrame:
    """
    Generate a DataFrame of airline data from contents of FleetData and CountryData files
    Within each region, the number of aircraft of each type assigned to each country is proportional to the country's GDP
    Within each country, the aircraft of each type are assigned to the airlines as evenly as possible

    Parameters
    ----------
    fleet_data : pd.DataFrame
    country_data : pd.DataFrame
    run_parameters : pd.DataFrame
    regions : list | None

    Returns
    -------
    airlines : pd.DataFrame
    """
    if ("CensusAsiaPacific" in fleet_data.columns) and ("CensusMiddleEast" in fleet_data.columns):
        # 2019 census format
        census_regions = {
            "Americas": [10, 11, 12],
            "Europe": [13],
            "MiddleEast": [14],
            "Africa": [15],
            "AsiaPacific": [16],
        }
    elif ("CensusAsiaAusMiddleEast" in fleet_data.columns):
        # 2015 census format
        census_regions = {
            "Americas": [10, 11, 12],
            "Europe": [13],
            "AsiaAusMiddleEast": [14, 16],
            "Africa": [15],
        }

    country_data["GDP"] = (
        country_data["PPP_GDP_Cap_Year2015USD_2015"]
        * country_data["Population_2015"]
    )
    country_data["Region_GDP_Proportion"] = 0.0  # initialise with zeros

    airlines = pd.DataFrame()
    airline_idx = 0
    n_aircraft_types = len(fleet_data)

    airline_id = []
    region_id = []
    country_name = []
    country_id = []
    aircraft_lists = []

    for region, codes in census_regions.items():
        # calculate GDP proportions
        countries = country_data["Region"].isin(codes)
        if not any(countries):
            continue
        region_gdp = country_data.loc[countries, "GDP"].sum()
        country_data.loc[countries, "Region_GDP_Proportion"] = (
            country_data.loc[countries, "GDP"] / region_gdp
        )

        # sort countries in region by GDP
        sorted_region = sorted(
            country_data.loc[countries].iterrows(),
            key=lambda x: (
                x[1]["BaseYearGDP"],
                x[1]["BaseYearPopulation"],
            ),  # Use population as tiebreaker
            reverse=True,
        )

        # assign aircraft to countries
        region_aircraft = fleet_data[f"Census{region}"].to_list()
        unallocated = region_aircraft.copy()
        for (_, country,) in sorted_region:
            country_aircraft = [0] * n_aircraft_types
            # iterate through aircraft types and account for rounding issues
            for i in range(n_aircraft_types):
                allocated = int(
                    round(region_aircraft[i] * country["Region_GDP_Proportion"])
                )
                if allocated == 0 and unallocated[i] > 0:
                    allocated = 1
                country_aircraft[i] = min(allocated, unallocated[i])
                unallocated[i] -= country_aircraft[i]

            # assign aircraft to airlines within country
            if regions is not None and country["Region"] not in regions:
                # no need to create multiple airlines for a country that won't be simulated
                n_airlines = 1
            elif sum(country_aircraft)/run_parameters["AirlinesPerCountry"] < MIN_INIT_PLANES_PER_AL:
                # prevent small countries from having lots of tiny airlines
                n_airlines = max(sum(country_aircraft) // MIN_INIT_PLANES_PER_AL, 1)
            else:
                n_airlines = run_parameters["AirlinesPerCountry"]
            for country_airline_idx in range(n_airlines):
                n_aircraft = [0] * n_aircraft_types
                for aircraft_type in range(n_aircraft_types):
                    base = (country_aircraft[aircraft_type] // n_airlines)
                    remainder = (country_aircraft[aircraft_type] % n_airlines)
                    # add one extra aircraft to each airline until the remainder is used up
                    n_aircraft[aircraft_type] = base + (
                        1 if country_airline_idx < remainder else 0
                    )

                # create airline, unless it has no aircraft
                if sum(n_aircraft) > 0:
                    airline_id.append(airline_idx)
                    region_id.append(country["Region"])
                    country_name.append(country["Country"])
                    country_id.append(country["Number"])
                    aircraft_lists.append(n_aircraft)
                    airline_idx += 1

    airlines["Airline_ID"] = airline_id
    airlines["Region"] = region_id
    airlines["Country"] = country_name
    airlines["CountryID"] = country_id
    airlines["n_Aircraft"] = aircraft_lists
    airlines["Grounded_acft"] = [[] for _ in range(len(airlines))]

    airlines.set_index("Airline_ID", inplace=True)

    return airlines


def initialise_fleet_assignment(
    airlines: pd.DataFrame,
    city_pair_data: pd.DataFrame,
    city_data: pd.DataFrame,
    aircraft_data: pd.DataFrame,
    city_lookup: list,
    randomGen: np.random.Generator,
    year: int,
    demand_coefficients: dict[str, float],
    regions: list | None,
) -> tuple[
    list[pd.DataFrame],
    list[pd.DataFrame],
    pd.DataFrame,
    pd.DataFrame,
    list[int]
]:
    """
    Assign aircraft to routes based on base demand, aircraft range and runway required
    Updates airlines and city_pair_data DataFrames in place.

    Parameters
    ----------
    airlines : pd.DataFrame
        dataframe containing airline data
    city_pair_data : pd.DataFrame
        dataframe containing route data
    city_data : pd.DataFrame
        dataframe containing city data
    aircraft_data : pd.DataFrame
        dataframe containing aircraft data
    city_lookup : list
        list of lists of city IDs by country code
    randomGen : np.random.Generator
        random number generator
    year : int
        calendar year
    demand_coefficients : dict
        dictionary of demand coefficients
    regions : list | None
        list of regions to include in the simulation. If None, all regions are included.
    
    Returns
    -------
    airline_fleets : list[pd.DataFrame]
    airline_routes : list[pd.DataFrame]
    city_pair_data : pd.DataFrame
    city_data : pd.DataFrame
    capacity_flag_list : list
    """
    # TODO: enable EU airlines to operate routes between all EU countries
    # TODO: move aircraft creation and following lines into a function to avoid code duplication

    n_airlines = len(airlines)

    # create a list of airline fleet DataFrames
    airline_fleets = [pd.DataFrame() for _ in range(n_airlines)]
    airline_routes = [pd.DataFrame(
        {
            "origin": pd.Series(dtype="int"),
            "destination": pd.Series(dtype="int"),
            "fuel_stop": pd.Series(dtype="int"),
            "fare": pd.Series(dtype="float"),
            "aircraft_ids": pd.Series(dtype="object"),
            "flights_per_year": pd.Series(dtype="int"),
            "seat_flights_per_year": pd.Series(dtype="int"),
            "exp_utility": pd.Series(dtype="float"),
        }
    ) for _ in range(n_airlines)]

    # iterate over all airlines
    print(f"    Assigning aircraft to {n_airlines} airlines...")
    i = 0
    for airline_id, airline in airlines.iterrows():
        if i % 10 == 0 and i > 0:
            print(f"        Airlines completed: {i}")
        i += 1

        # calculate total base RPKs for all routes the airline can operate (assume airlines can only run routes to/from their home country)
        possible_RPKs = 0.0
        distances = []
        for origin_id in city_lookup[airline["CountryID"]]:
            for destination_id in city_data.index.to_list():
                if not (origin_id == destination_id):
                    # check outbound and inbound routes exist in city_pair_data
                    if (not city_pair_data[
                        (city_pair_data["OriginCityID"] == origin_id)
                        & (city_pair_data["DestinationCityID"] == destination_id)
                    ].empty) and (not city_pair_data[
                        (city_pair_data["OriginCityID"] == destination_id)
                        & (city_pair_data["DestinationCityID"] == origin_id)
                    ].empty):
                        outbound_route = city_pair_data[
                            (city_pair_data["OriginCityID"] == origin_id)
                            & (city_pair_data["DestinationCityID"] == destination_id)
                        ].iloc[0]
                        inbound_route = city_pair_data[
                            (city_pair_data["OriginCityID"] == destination_id)
                            & (city_pair_data["DestinationCityID"] == origin_id)
                        ].iloc[0]

                        route_RPKs = (
                            outbound_route["BaseYearODDemandPax_Est"]
                            * outbound_route["Great_Circle_Distance_m"]
                        )
                        route_RPKs += (
                            inbound_route["BaseYearODDemandPax_Est"]
                            * inbound_route["Great_Circle_Distance_m"]
                        )
                        possible_RPKs += route_RPKs
                        # append a tuple (origin_id, destination_id, distance, RPKs) to distances list
                        distances.append(
                            (
                                origin_id,
                                destination_id,
                                outbound_route["Great_Circle_Distance_m"],
                                route_RPKs,
                            )
                        )

        # calculate total airline seat capacity
        total_seats = 0
        smallest_ac = np.inf
        for aircraft_type, n_aircraft_type in enumerate(airline["n_Aircraft"]):
            total_seats += n_aircraft_type * aircraft_data.loc[aircraft_type, "Seats"]
            if (n_aircraft_type > 0) and (aircraft_data.loc[aircraft_type, "Seats"] < smallest_ac):
                smallest_ac = aircraft_data.loc[aircraft_type, "Seats"]

        aircraft_id_list = []
        aircraft_size_list = []
        aircraft_age_list = []
        current_lease_list = []
        breguet_factor_list = []
        origin_id_list = []
        destination_id_list = []
        fuel_stop_list = []
        flights_per_year_list = []
        capacity_flag_list = []

        # assign aircraft seat capacity by base demand starting with the largest aircraft on the longest routes
        distances.sort(key=lambda x: x[2], reverse=True)
        aircraft_avail = airline["n_Aircraft"].copy()
        aircraft_id = -1
        for origin_id, destination_id, distance, route_RPKs in distances:
            # stop if no aircraft available
            if sum(aircraft_avail) == 0:
                break
            
            outbound_route = city_pair_data[
                (city_pair_data["OriginCityID"] == origin_id)
                & (city_pair_data["DestinationCityID"] == destination_id)
            ].iloc[0]
            inbound_route = city_pair_data[
                (city_pair_data["OriginCityID"] == destination_id)
                & (city_pair_data["DestinationCityID"] == origin_id)
            ].iloc[0]
            
            seats = total_seats * (route_RPKs / possible_RPKs)

            for aircraft_size, aircraft in aircraft_data.iterrows():
                if seats <= (smallest_ac * MIN_PAX_LOAD_FACTOR):
                    break
                if aircraft_avail[aircraft_size] > 0:
                    # check origin and destination runways are long enough
                    if (
                        city_data.loc[origin_id, "LongestRunway_m"] > aircraft["TakeoffDist_m"]
                        and city_data.loc[origin_id, "LongestRunway_m"] > aircraft["LandingDist_m"]
                        and city_data.loc[destination_id, "LongestRunway_m"] > aircraft["TakeoffDist_m"]
                        and city_data.loc[destination_id, "LongestRunway_m"] > aircraft["LandingDist_m"]
                    ):
                        # check aircraft has enough range
                        if distance < aircraft["TypicalRange_m"]:  # must be kept seperate due to else statement
                            while(
                                (seats / aircraft["Seats"] > MIN_PAX_LOAD_FACTOR)
                                and (aircraft_avail[aircraft_size] > 0)
                            ):
                                seats -= aircraft["Seats"]  # can go negative
                                aircraft_avail[aircraft_size] -= 1

                                # create aircraft and assign to route, randomise age
                                aircraft_id += 1
                                fuel_stop = -1

                                (
                                    aircraft_id_list,
                                    aircraft_size_list,
                                    aircraft_age_list,
                                    current_lease_list,
                                    breguet_factor_list,
                                    origin_id_list,
                                    destination_id_list,
                                    fuel_stop_list,
                                    flights_per_year_list,
                                    city_pair_data,
                                    city_data,
                                    airline_routes,
                                    capacity_flag_list,
                                ) = create_aircraft(
                                    aircraft_id_list,
                                    aircraft_id,
                                    aircraft_size_list,
                                    aircraft_size,
                                    aircraft_age_list,
                                    randomGen,
                                    aircraft,
                                    current_lease_list,
                                    breguet_factor_list,
                                    year,
                                    origin_id_list,
                                    origin_id,
                                    destination_id_list,
                                    destination_id,
                                    fuel_stop_list,
                                    fuel_stop,
                                    flights_per_year_list,
                                    city_data,
                                    city_pair_data,
                                    airline_routes,
                                    airline_id,
                                    outbound_route,
                                    inbound_route,
                                    capacity_flag_list,
                                    regions,
                                )

                        elif distance < 2*aircraft["TypicalRange_m"]:
                            # aircraft doesn't have enough range - try with a fuel stop
                            fuel_stop = route.choose_fuel_stop(
                                city_data,
                                city_data.loc[origin_id],
                                city_data.loc[destination_id],
                                aircraft["TypicalRange_m"],
                                min(
                                    aircraft["TakeoffDist_m"],
                                    aircraft["LandingDist_m"]
                                ),
                                regions,
                            )

                            if not (fuel_stop == -1):  # -1 if route not possible with a stop for that aircraft
                                while(
                                    (seats / aircraft["Seats"] > MIN_PAX_LOAD_FACTOR)
                                    and (aircraft_avail[aircraft_size] > 0)
                                ):
                                    seats -= aircraft["Seats"]  # can go negative
                                    aircraft_avail[aircraft_size] -= 1

                                    # create aircraft instance and assign to route, randomise age
                                    aircraft_id += 1

                                    (
                                        aircraft_id_list,
                                        aircraft_size_list,
                                        aircraft_age_list,
                                        current_lease_list,
                                        breguet_factor_list,
                                        origin_id_list,
                                        destination_id_list,
                                        fuel_stop_list,
                                        flights_per_year_list,
                                        city_pair_data,
                                        city_data,
                                        airline_routes,
                                        capacity_flag_list,
                                    ) = create_aircraft(
                                        aircraft_id_list,
                                        aircraft_id,
                                        aircraft_size_list,
                                        aircraft_size,
                                        aircraft_age_list,
                                        randomGen,
                                        aircraft,
                                        current_lease_list,
                                        breguet_factor_list,
                                        year,
                                        origin_id_list,
                                        origin_id,
                                        destination_id_list,
                                        destination_id,
                                        fuel_stop_list,
                                        fuel_stop,
                                        flights_per_year_list,
                                        city_data,
                                        city_pair_data,
                                        airline_routes,
                                        airline_id,
                                        outbound_route,
                                        inbound_route,
                                        capacity_flag_list,
                                        regions,
                                    )

                        else:
                            break
        
        fleet_df = pd.DataFrame()

        fleet_df["AircraftID"] = np.array(aircraft_id_list).astype(int)
        fleet_df["SizeClass"] = np.array(aircraft_size_list).astype(int)
        fleet_df["Age_years"] = np.array(aircraft_age_list).astype(int)
        fleet_df["Lease_USDperMonth"] = current_lease_list
        fleet_df["BreguetFactor"] = breguet_factor_list
        fleet_df["RouteOrigin"] = np.array(origin_id_list).astype(int)
        fleet_df["RouteDestination"] = np.array(destination_id_list).astype(int)
        fleet_df["FuelStop"] = np.array(fuel_stop_list).astype(int)
        fleet_df["Flights_perYear"] = np.array(flights_per_year_list).astype(int)

        # calculate exp(utility) now that mean travel times can be calculated
        airline_routes, city_pair_data = recalculate_exp_utility(
            airline_routes,
            airline_id,
            demand_coefficients,
            city_data,
            city_pair_data,
            aircraft_data,
            fleet_df,
        )

        airline_fleets[airline_id] = fleet_df

    return airline_fleets, airline_routes, city_pair_data, city_data, capacity_flag_list


def create_aircraft(
    aircraft_id_list: list,
    aircraft_id: int,
    aircraft_size_list: list,
    aircraft_size: int,
    aircraft_age_list: list,
    randomGen: np.random.Generator,
    aircraft: pd.Series,
    current_lease_list: list,
    breguet_factor_list: list,
    year: int,
    origin_id_list: list,
    origin_id: int,
    destination_id_list: list,
    destination_id: int,
    fuel_stop_list: list,
    fuel_stop: int,
    flights_per_year_list: list,
    city_data: pd.DataFrame,
    city_pair_data: pd.DataFrame,
    airline_routes: list[pd.DataFrame],
    airline_id: int,
    outbound_route: pd.Series,
    inbound_route: pd.Series,
    capacity_flag_list: list,
    regions: list | None,
):
    if fuel_stop == -1:
        fuel_stop_series = None
    else:
        fuel_stop_series = city_data.loc[fuel_stop]

    flights_per_year = acft.calc_flights_per_year(
        city_data.loc[origin_id],
        origin_id,
        city_data.loc[destination_id],
        destination_id,
        aircraft,
        city_pair_data,
        fuel_stop_series,
        fuel_stop
    )

    if (
        regions is None
        or (
            city_data.loc[origin_id, "Region"] in regions
            and city_data.loc[destination_id, "Region"] in regions
        )
    ):
        # simulating the whole world or the itinerary is contained entirely within the simulated region

        flights_per_year_list.append(flights_per_year)

        # check if route already exists in airline_routes
        not_route_exists = airline_routes[airline_id][
            (airline_routes[airline_id]["origin"] == origin_id)
            & (airline_routes[airline_id]["destination"] == destination_id)
            & (airline_routes[airline_id]["fuel_stop"] == fuel_stop)  # treat as a seperate itinerary if fuel stop is different
        ].empty

        aircraft_id_list.append(aircraft_id)
        aircraft_size_list.append(aircraft_size)
        aircraft_age_list.append(randomGen.randint(0, aircraft["RetirementAge_years"]-1))
        current_lease_list.append(aircraft["LeaseRateNew_USDPerMonth"] * (aircraft["LeaseRateAnnualMultiplier"] ** aircraft_age_list[-1]))
        breguet_factor_list.append((aircraft["Breguet_gradient"] * year) + aircraft["Breguet_intercept"])
        origin_id_list.append(origin_id)
        destination_id_list.append(destination_id)
        fuel_stop_list.append(fuel_stop)

        # add movements to cities and flag if capacity limit exceeded
        city_data.loc[origin_id, "Movts_perHr"] += 2.0 * float(flights_per_year) / OP_HRS_PER_YEAR  # 2* since each flight is return
        city_data.loc[destination_id, "Movts_perHr"] += 2.0 * float(flights_per_year) / OP_HRS_PER_YEAR
        
        if (
            (city_data.loc[origin_id, "Movts_perHr"] > city_data.loc[origin_id, "Capacity_MovtsPerHr"])
            and (origin_id not in capacity_flag_list)
        ):
            capacity_flag_list.append(origin_id)
        if (
            (city_data.loc[destination_id, "Movts_perHr"] > city_data.loc[destination_id, "Capacity_MovtsPerHr"])
            and (destination_id not in capacity_flag_list)
        ):
            capacity_flag_list.append(destination_id)

        if fuel_stop != -1:
            city_data.loc[fuel_stop, "Movts_perHr"] += 4.0 * float(flights_per_year) / OP_HRS_PER_YEAR  # 4* since 1 takeoff and 1 landing for each leg
            if (
                (city_data.loc[fuel_stop, "Movts_perHr"] > city_data.loc[fuel_stop, "Capacity_MovtsPerHr"])
                and (fuel_stop not in capacity_flag_list)
            ):
                capacity_flag_list.append(fuel_stop)

        seat_flights_per_year = flights_per_year * aircraft["Seats"]

        if not not_route_exists:
            # airline already has aircraft assigned to this route
            outbound_mask = (
                (airline_routes[airline_id]["origin"] == origin_id)
                & (airline_routes[airline_id]["destination"] == destination_id)
                & (airline_routes[airline_id]["fuel_stop"] == fuel_stop)
            )
            inbound_mask = (
                (airline_routes[airline_id]["origin"] == destination_id)
                & (airline_routes[airline_id]["destination"] == origin_id)
                & (airline_routes[airline_id]["fuel_stop"] == fuel_stop)
            )

        # update outbound and return route in-place in city_pair_data DataFrame
        city_pair_data.loc[
            (city_pair_data["OriginCityID"] == origin_id)
            & (city_pair_data["DestinationCityID"] == destination_id),
            "Seat_Flights_perYear"
        ] += seat_flights_per_year
        city_pair_data.loc[
            (city_pair_data["OriginCityID"] == destination_id)
            & (city_pair_data["DestinationCityID"] == origin_id),
            "Seat_Flights_perYear"
        ] += seat_flights_per_year

        # update airline-specific route dataframe
        if not_route_exists:
            airline_routes_newrow_1 = {
                "origin": [origin_id],
                "destination": [destination_id],
                "fare": [outbound_route["Fare_Est"]],
                "aircraft_ids": [[aircraft_id]],
                "flights_per_year": [flights_per_year],
                "seat_flights_per_year": [seat_flights_per_year],
                "exp_utility": [0.0],  # calculated later
                "fuel_stop": [fuel_stop]
            }
            new_df_1 = pd.DataFrame(airline_routes_newrow_1)
            airline_routes_newrow_2 = {
                "origin": [destination_id],
                "destination": [origin_id],
                "fare": [inbound_route["Fare_Est"]],
                "aircraft_ids": [[aircraft_id]],
                "flights_per_year": [flights_per_year],
                "seat_flights_per_year": [seat_flights_per_year],
                "exp_utility": [0.0],  # calculated later
                "fuel_stop": [fuel_stop]
            }
            new_df_2 = pd.DataFrame(airline_routes_newrow_2)
            airline_routes[airline_id] = pd.concat([
                airline_routes[airline_id], 
                new_df_1,
                new_df_2
            ], ignore_index=True)
        else:
            airline_routes[airline_id].loc[outbound_mask, "aircraft_ids"].iloc[0].append(aircraft_id)
            airline_routes[airline_id].loc[inbound_mask, "aircraft_ids"].iloc[0].append(aircraft_id)

            airline_routes[airline_id].loc[outbound_mask, "flights_per_year"] += flights_per_year
            airline_routes[airline_id].loc[inbound_mask, "flights_per_year"] += flights_per_year

            airline_routes[airline_id].loc[outbound_mask, "seat_flights_per_year"] += seat_flights_per_year
            airline_routes[airline_id].loc[inbound_mask, "seat_flights_per_year"] += seat_flights_per_year

    elif (
        city_data.loc[origin_id, "Region"] in regions
        or city_data.loc[destination_id, "Region"] in regions
        or (fuel_stop != -1 and fuel_stop_series["Region"] in regions)
    ):
        # itinerary exists partially within the simulated region, so we care about movements only

        # add movements to cities and flag if capacity limit exceeded
        if city_data.loc[origin_id, "Region"] in regions:
            movts = 2.0 * float(flights_per_year) / OP_HRS_PER_YEAR  # 2* since each flight is return
            city_data.loc[origin_id, "Movts_perHr"] += movts
            city_data.loc[origin_id, "Movts_Outside"] += movts
            if (
                (city_data.loc[origin_id, "Movts_perHr"] > city_data.loc[origin_id, "Capacity_MovtsPerHr"])
                and (origin_id not in capacity_flag_list)
            ):
                capacity_flag_list.append(origin_id)

        if city_data.loc[destination_id, "Region"] in regions:
            movts = 2.0 * float(flights_per_year) / OP_HRS_PER_YEAR
            city_data.loc[destination_id, "Movts_perHr"] += movts
            city_data.loc[destination_id, "Movts_Outside"] += movts
            if (
                (city_data.loc[destination_id, "Movts_perHr"] > city_data.loc[destination_id, "Capacity_MovtsPerHr"])
                and (destination_id not in capacity_flag_list)
            ):
                capacity_flag_list.append(destination_id)
        
        if fuel_stop != -1:
            if fuel_stop_series["Region"] in regions:
                movts = 4.0 * float(flights_per_year) / OP_HRS_PER_YEAR  # 4* since 1 takeoff and 1 landing for each leg
                city_data.loc[fuel_stop, "Movts_perHr"] += movts
                city_data.loc[fuel_stop, "Movts_Outside"] += movts
                if (
                    (city_data.loc[fuel_stop, "Movts_perHr"] > city_data.loc[fuel_stop, "Capacity_MovtsPerHr"])
                    and (fuel_stop not in capacity_flag_list)
                ):
                    capacity_flag_list.append(fuel_stop)
    
    # else:
        # itinerary exists entirely outside the simulated region => do nothing

    return (
        aircraft_id_list,
        aircraft_size_list,
        aircraft_age_list,
        current_lease_list,
        breguet_factor_list,
        origin_id_list,
        destination_id_list,
        fuel_stop_list,
        flights_per_year_list,
        city_pair_data,
        city_data,
        airline_routes,
        capacity_flag_list,
    )


def recalculate_exp_utility(
    airline_routes: list[pd.DataFrame],
    airline_id: int,
    demand_coefficients: dict[str, float],
    city_data: pd.DataFrame,
    city_pair_data: pd.DataFrame,
    aircraft_data: pd.DataFrame,
    fleet_data: pd.DataFrame,
) -> list[pd.DataFrame]:
    for itin_idx, itin in airline_routes[airline_id].iterrows():
        fare = airline_routes[airline_id].loc[itin_idx, "fare"]
        flights_per_year = airline_routes[airline_id].loc[itin_idx, "flights_per_year"]
        fuel_stop = airline_routes[airline_id].loc[itin_idx, "fuel_stop"]

        itin_time_hrs = acft.calc_itin_time(
            itin,
            city_data,
            city_pair_data,
            aircraft_data,
            fleet_data,
        )

        airline_routes[airline_id].loc[itin_idx, "exp_utility"] = demand.calc_exp_utility(
            demand_coefficients,
            fare,
            itin_time_hrs,
            flights_per_year,
            fuel_stop,
        )

        # add itinerary contribution to total exp(utility) in city_pair_data
        city_pair_data.loc[
            (city_pair_data["OriginCityID"] == itin["origin"])
            & (city_pair_data["DestinationCityID"] == itin["destination"]),
            "Exp_Utility_Sum"
        ] += airline_routes[airline_id].loc[itin_idx, "exp_utility"]

    return airline_routes, city_pair_data


def optimise_fares(
    airlines: pd.DataFrame,
    airline_routes: list[pd.DataFrame],
    airline_fleets: list[pd.DataFrame],
    city_pair_data: pd.DataFrame,
    city_data: pd.DataFrame,
    aircraft_data: pd.DataFrame,
    max_fare: float,
    FuelCost_USDperGallon,
    demand_coefficients: dict[str, float],
) -> tuple[
    list[pd.DataFrame],
    pd.DataFrame
]:
    """
    Airlines adjust their fares to attempt to maximise profit, assuming no change in flight schedules.
    Airlines have no knowledge of the choices of other airlines.

    Parameters
    ----------
    airlines : pd.DataFrame
    airline_routes : list[pd.DataFrame]
    airline_fleets : list[pd.DataFrame]
    city_pair_data : pd.DataFrame
    city_data : pd.DataFrame
    aircraft_data : pd.DataFrame
    max_fare : float
    FuelCost_USDperGallon : float

    Returns
    -------
    airline_routes : list[pd.DataFrame]
    city_pair_data : pd.DataFrame
    """
    city_pair_data["New_Mean_Fare_USD"] = city_pair_data["Mean_Fare_USD"].copy()
    city_pair_data["New_Exp_Utility_Sum"] = city_pair_data["Exp_Utility_Sum"].copy()

    print(f"        Optimising fares...")
    print("        Time: ", datetime.datetime.now(), "\n")
    for airline_id in airlines.index:
        airline_fleet_data = airline_fleets[airline_id]
        airline_route_data = airline_routes[airline_id]

        for idx, itin in airline_route_data.iterrows():
            city_pair = city_pair_data[
                (city_pair_data["OriginCityID"] == itin["origin"])
                & (city_pair_data["DestinationCityID"] == itin["destination"])
            ].iloc[0]
            old_fare = itin["fare"]
            old_exp_utility = itin["exp_utility"]
            new_fare = maximise_itin_profit(
                itin,
                airline_fleet_data,
                city_pair_data,
                city_data,
                aircraft_data,
                max_fare,
                FuelCost_USDperGallon,
                demand_coefficients,
            )
            new_exp_utility = demand.calc_exp_utility(
                demand_coefficients,
                new_fare,
                itin["itin_time_hrs"],
                itin["flights_per_year"],
                itin["fuel_stop"],
            )
            airline_routes[airline_id].loc[idx, "fare"] = new_fare
            airline_routes[airline_id].loc[idx, "exp_utility"] = new_exp_utility

            # save new mean fare and exp(utility) for updating city_pair_data later
            # (don't edit in-place because airlines shouldn't know the decisions of other airlines)
            itin_fare_diff = new_fare - old_fare
            itin_exp_utility_diff = new_exp_utility - old_exp_utility
            prev_mean_fare = city_pair["New_Mean_Fare_USD"]
            prev_exp_utility_sum = city_pair["New_Exp_Utility_Sum"]
            city_pair_data.loc[
                (city_pair_data["OriginCityID"] == itin["origin"])
                & (city_pair_data["DestinationCityID"] == itin["destination"]),
                "New_Mean_Fare_USD"
            ] = (
                (prev_mean_fare * city_pair["Seat_Flights_perYear"])
                + (itin_fare_diff * itin["seat_flights_per_year"])
            ) / city_pair["Seat_Flights_perYear"]
            city_pair_data.loc[
                (city_pair_data["OriginCityID"] == itin["origin"])
                & (city_pair_data["DestinationCityID"] == itin["destination"]),
                "New_Exp_Utility_Sum"
            ] = (prev_exp_utility_sum + itin_exp_utility_diff)

    # update city_pair_data mean fare
    city_pair_data["Mean_Fare_USD"] = city_pair_data["New_Mean_Fare_USD"].copy()
    city_pair_data["Exp_Utility_Sum"] = city_pair_data["New_Exp_Utility_Sum"].copy()

    # update demand for all O-D pairs
    for idx, city_pair in city_pair_data.iterrows():
        city_pair_data.loc[idx, "Total_Demand"] = demand.update_od_demand(city_pair)

    return airline_routes, city_pair_data


def maximise_itin_profit(
    airline_route: pd.Series,
    fleet_df: pd.DataFrame,
    city_pair_data: pd.DataFrame,
    city_data: pd.DataFrame,
    aircraft_data: pd.DataFrame,
    max_fare: float,
    FuelCost_USDperGallon: float,
    demand_coefficients: dict[str, float],
) -> float:
    """
    Maximise profit for a given itinerary by adjusting fare
    """
    origin = city_data.loc[airline_route["origin"]]
    destination = city_data.loc[airline_route["destination"]]
    city_pair = city_pair_data[
        (city_pair_data["OriginCityID"] == airline_route["origin"])
        & (city_pair_data["DestinationCityID"] == airline_route["destination"])
    ].iloc[0]

    fare_bounds = (1.0, max_fare)

    # Minimise negative profit
    def objective(fare):
        return -itin_profit(
            airline_route,
            city_pair,
            origin,
            destination,
            fleet_df,
            aircraft_data,
            FuelCost_USDperGallon,
            demand_coefficients,
            new_itin_fare=fare[0],
        )
    
    initial_fare = max([min(city_pair["Mean_Fare_USD"], max_fare),1.0])

    result = gp_minimize(
        objective,
        [fare_bounds],
        n_calls=10,
        n_initial_points=3,
        x0=[[initial_fare]],
        y0=[objective([initial_fare])],
        n_jobs=1,  # force single thread because dataframes are modified in-place
    )
    optimal_fare = float(result.x[0])
    return optimal_fare


def itin_profit(
    airline_route: pd.Series,
    city_pair_in: pd.Series,
    origin: pd.Series,
    destination: pd.Series,
    fleet_data: pd.DataFrame,
    aircraft_data: pd.DataFrame,
    FuelCost_USDperGallon: float,
    demand_coefficients: dict[str, float],
    new_itin_fare: float | None = None,  # new airline-specific itinerary fare to test
) -> float:
    """
    Calculate profit for a given itinerary (one-way) and fare
    """
    city_pair = city_pair_in.copy()  # copy explicitly to avoid SettingWithCopyWarning
    if new_itin_fare is not None:
        old_itin_fare = airline_route["fare"]
        old_itin_exp_utility = airline_route["exp_utility"]
        old_od_exp_utility_sum = city_pair["Exp_Utility_Sum"]
        old_od_mean_fare = city_pair["Mean_Fare_USD"]
        airline_route["fare"] = new_itin_fare

        # update city_pair mean fare (weighted by seats available not seats filled)
        total_revenue = city_pair["Mean_Fare_USD"] * city_pair["Seat_Flights_perYear"]
        total_revenue += ((new_itin_fare - old_itin_fare) * airline_route["seat_flights_per_year"])
        city_pair["Mean_Fare_USD"] = total_revenue / city_pair["Seat_Flights_perYear"]

        # update variables that determine itinerary market share
        airline_route["exp_utility"] = demand.calc_exp_utility(
            demand_coefficients,
            new_itin_fare,
            airline_route["itin_time_hrs"],
            airline_route["flights_per_year"],
            airline_route["fuel_stop"],
        )
        city_pair["Exp_Utility_Sum"] = (
            old_od_exp_utility_sum
            + airline_route["exp_utility"]
            - old_itin_exp_utility
        )

    # update OD demand
    old_total_demand = city_pair["Total_Demand"]
    city_pair["Total_Demand"] = demand.update_od_demand(city_pair)

    # calculate tickets sold (can't be more than the airline has scheduled or less than zero)
    annual_itin_demand = demand.update_itinerary_demand(city_pair, airline_route)
    tickets_sold = min([annual_itin_demand, airline_route["seat_flights_per_year"]])
    tickets_sold = max([0, tickets_sold])
    
    annual_itin_revenue = tickets_sold * airline_route["fare"]

    annual_itin_cost = acft.calc_flight_cost(
        airline_route,
        fleet_data,
        aircraft_data,
        city_pair,
        origin,
        destination,
        tickets_sold,
        FuelCost_USDperGallon,
    )  # note depends on number of seats sold due to weight
    annual_itin_profit = annual_itin_revenue - annual_itin_cost

    # reset dataframes to avoid mutability issues
    city_pair["Total_Demand"] = old_total_demand
    if new_itin_fare is not None:
        airline_route["fare"] = old_itin_fare
        city_pair["Mean_Fare_USD"] = old_od_mean_fare
        airline_route["exp_utility"] = old_itin_exp_utility
        city_pair["Exp_Utility_Sum"] = old_od_exp_utility_sum
    
    return annual_itin_profit


def reassign_ac_for_profit(
    airlines: pd.DataFrame,
    airline_routes: list[pd.DataFrame],
    airline_fleets: list[pd.DataFrame],
    city_pair_data: pd.DataFrame,
    city_data: pd.DataFrame,
    city_lookup: list,
    aircraft_data: pd.DataFrame,
    demand_coefficients: dict[str, float],
    FuelCost_USDperGallon: float,
    year: int,
) -> tuple[
    list[pd.DataFrame],
    list[pd.DataFrame],
    pd.DataFrame,
    pd.DataFrame
]:
    """
    Reassign aircraft to different routes to maximise profit per seat
    
    - Airline by airline, starts with the least profitable route and reassigns aircraft to the most profitable alternative route
    - Stops when the least profitable aircraft can't be reassigned to a more profitable route
    - If an aircraft is making negative profit and can't be reassigned to a more profitable route, it is grounded
    - If an aircraft grounded in the previous year can't be assigned to a profitable route, the lease is ended
    - If an airline has no grounded aircraft the previous year and current year, it has the opportunity to lease new aircraft

    Parameters
    ----------
    airlines : pd.DataFrame
    airline_routes : list[pd.DataFrame]
    airline_fleets : list[pd.DataFrame]
    city_pair_data : pd.DataFrame
    city_data : pd.DataFrame
    city_lookup : list
    aircraft_data : pd.DataFrame
    demand_coefficients : dict
    FuelCost_USDperGallon : float

    Returns
    -------
    airline_routes : list[pd.DataFrame]
    airline_fleets : list[pd.DataFrame]
    city_pair_data : pd.DataFrame
    city_data : pd.DataFrame
    """
    # TODO: - address the issue of the order of airlines giving some an advantage over others
    #       - check each size class seperately
    #       - consider moving aircraft to routes with fuel stops

    print(f"        Reassigning aircraft...")
    print("        Time: ", datetime.datetime.now(), "\n")
    # iterate over all airlines
    for airline_id, airline in airlines.iterrows():
        # create dataframe of airline's existing routes and their profit per seat
        rtn_flt_df = reassignment.calc_existing_profits(
            airline_routes[airline_id],
            city_data,
            city_pair_data,
            aircraft_data,
            airline_fleets[airline_id],
            FuelCost_USDperGallon,
            demand_coefficients,
        )

        # reassign aircraft one-by-one to most profitable route and stop when no more profitable routes are available
        finished = False
        while not finished:
            if len(rtn_flt_df) == 0:
                finished = True
                break
            
            reassign_row = rtn_flt_df.loc[rtn_flt_df["Profit_perSeat"].idxmin()]  # least profitable itinerary
            reassign_itin_out = airline_routes[airline_id].loc[
                (airline_routes[airline_id]["origin"] == reassign_row["Origin"])
                & (airline_routes[airline_id]["destination"] == reassign_row["Destination"])
                & (airline_routes[airline_id]["fuel_stop"] == reassign_row["Fuel_Stop"])
            ].iloc[0]
            reassign_itin_in = airline_routes[airline_id].loc[
                (airline_routes[airline_id]["origin"] == reassign_row["Destination"])
                & (airline_routes[airline_id]["destination"] == reassign_row["Origin"])
                & (airline_routes[airline_id]["fuel_stop"] == reassign_row["Fuel_Stop"])
            ].iloc[0]
            reassign_old_profit_per_seat = reassign_row["Profit_perSeat"]

            # choose largest aircraft on route to reassign
            itin_aircraft_ids = reassign_row["Aircraft_IDs"]
            itin_ac = airline_fleets[airline_id][airline_fleets[airline_id]["AircraftID"].isin(itin_aircraft_ids)].copy()
            itin_ac.sort_values(["SizeClass", "Age_years"], ascending=[False, False], inplace=True)
            reassign_ac = itin_ac.iloc[0]  # largest aircraft only

            # find new profit per seat of route that aircraft is being removed from
            reassign_new_profit_per_seat = reassignment.profit_after_removal(
                itin_ac,
                reassign_ac,
                airline_fleets[airline_id],
                reassign_itin_out,
                reassign_itin_in,
                aircraft_data,
                city_data,
                city_pair_data,
                demand_coefficients,
                FuelCost_USDperGallon,
            )

            # test new itineraries one-by-one and save the results from the most profitable
            (
                delta_profit_per_seat,
                new_origin, new_destination,
                addnl_flights_per_year,
                addnl_seat_flights_per_year,
            ) = reassignment.find_itin_alternative(
                city_data,
                city_pair_data,
                city_lookup,
                airline["CountryID"],
                airline_routes[airline_id],
                airline_fleets[airline_id],
                aircraft_data,
                reassign_ac,
                demand_coefficients,
                FuelCost_USDperGallon,
                reassign_new_profit_per_seat,
                reassign_old_profit_per_seat,
            )

            # assign the aircraft to the most profitable alternative if beneficial
            if delta_profit_per_seat > 0.0:
                out_reassign_mask = (
                    (airline_routes[airline_id]["origin"] == reassign_itin_out["origin"])
                    & (airline_routes[airline_id]["destination"] == reassign_itin_out["destination"])
                    & (airline_routes[airline_id]["fuel_stop"] == reassign_itin_out["fuel_stop"])
                )
                in_reassign_mask = (
                    (airline_routes[airline_id]["origin"] == reassign_itin_in["origin"])
                    & (airline_routes[airline_id]["destination"] == reassign_itin_in["destination"])
                    & (airline_routes[airline_id]["fuel_stop"] == reassign_itin_in["fuel_stop"])
                )

                (
                    airlines,
                    airline_routes,
                    airline_fleets,
                    city_data,
                    city_pair_data,
                ) = reassignment.reassign_ac_to_new_route(
                    new_origin,
                    new_destination,
                    out_reassign_mask,
                    in_reassign_mask,
                    reassign_itin_out,
                    reassign_itin_in,
                    reassign_ac,
                    addnl_flights_per_year,
                    addnl_seat_flights_per_year,
                    city_pair_data,
                    city_data,
                    airline_routes,
                    airlines,
                    airline_id,
                    airline_fleets,
                    aircraft_data,
                    demand_coefficients,
                )

                # update masks because an itinerary may have been added to airline_routes
                out_reassign_mask = (
                    (airline_routes[airline_id]["origin"] == reassign_itin_out["origin"])
                    & (airline_routes[airline_id]["destination"] == reassign_itin_out["destination"])
                    & (airline_routes[airline_id]["fuel_stop"] == reassign_itin_out["fuel_stop"])
                )
                in_reassign_mask = (
                    (airline_routes[airline_id]["origin"] == reassign_itin_in["origin"])
                    & (airline_routes[airline_id]["destination"] == reassign_itin_in["destination"])
                    & (airline_routes[airline_id]["fuel_stop"] == reassign_itin_in["fuel_stop"])
                )
                # update rtn_flt_df
                rtn_flt_df = reassignment.update_profit_tracker(
                    rtn_flt_df,
                    new_origin,
                    new_destination,
                    out_reassign_mask,
                    in_reassign_mask,
                    reassign_itin_out,
                    reassign_ac,
                    airline_routes,
                    airline_fleets,
                    airline_id,
                    city_pair_data,
                    city_data,
                    aircraft_data,
                    FuelCost_USDperGallon,
                    demand_coefficients,
                )

                # if no planes left, remove itinerary from airline_routes and rtn_flt_df
                if len(airline_routes[airline_id].loc[out_reassign_mask, "aircraft_ids"].iloc[0]) == 0:
                    # remove itinerary from airline_routes
                    airline_routes[airline_id] = airline_routes[airline_id].loc[~out_reassign_mask]
                    in_reassign_mask = (
                        (airline_routes[airline_id]["origin"] == reassign_itin_in["origin"])
                        & (airline_routes[airline_id]["destination"] == reassign_itin_in["destination"])
                        & (airline_routes[airline_id]["fuel_stop"] == reassign_itin_in["fuel_stop"])
                    )  # recalculate mask since outbound itinerary has been removed
                    airline_routes[airline_id] = airline_routes[airline_id].loc[~in_reassign_mask]
                    # remove itinerary from rtn_flt_df
                    rtn_flt_df = rtn_flt_df.drop(
                        rtn_flt_df[
                            (rtn_flt_df["Origin"] == reassign_itin_out["origin"]) &
                            (rtn_flt_df["Destination"] == reassign_itin_out["destination"]) &
                            (rtn_flt_df["Fuel_Stop"] == reassign_itin_out["fuel_stop"])
                        ].index
                    )
            else:
                # if no beneficial change can be made, finished = True
                finished = True
            
        # deal with grounding and leases where appropriate
        rtn_flt_df.sort_values("Profit_perSeat", ascending=True, inplace=True)
        # check whether any itineraries are making a loss
        if (
            len(rtn_flt_df) > 0
            and rtn_flt_df["Profit_perSeat"].iloc[0] < 0.0
        ):
            # end lease on all currently grounded aircraft
            for ac_idx in airlines.loc[airline_id, "Grounded_acft"]:
                # remove aircraft from airline["n_Aircraft"]
                sizeclass = int(airline_fleets[airline_id].loc[
                    airline_fleets[airline_id]["AircraftID"] == ac_idx, "SizeClass"
                ].iloc[0])
                airlines.loc[airline_id, "n_Aircraft"][sizeclass] -= 1

                # remove aircraft from airline_fleets
                airline_fleets[airline_id] = airline_fleets[airline_id][
                    airline_fleets[airline_id]["AircraftID"] != ac_idx
                ]
            airlines.at[airline_id, "Grounded_acft"] = []

            # ground aircraft that are making a loss
            while (
                len(rtn_flt_df) > 0
                and rtn_flt_df["Profit_perSeat"].iloc[0] < 0.0
            ):
                reassign_row = rtn_flt_df.iloc[0]  # least profitable itinerary
                reassign_itin_out = airline_routes[airline_id].loc[
                    (airline_routes[airline_id]["origin"] == reassign_row["Origin"])
                    & (airline_routes[airline_id]["destination"] == reassign_row["Destination"])
                    & (airline_routes[airline_id]["fuel_stop"] == reassign_row["Fuel_Stop"])
                ].iloc[0]
                reassign_itin_in = airline_routes[airline_id].loc[
                    (airline_routes[airline_id]["origin"] == reassign_row["Destination"])
                    & (airline_routes[airline_id]["destination"] == reassign_row["Origin"])
                    & (airline_routes[airline_id]["fuel_stop"] == reassign_row["Fuel_Stop"])
                ].iloc[0]
                reassign_old_profit_per_seat = reassign_row["Profit_perSeat"]

                # choose largest aircraft on route to ground
                itin_aircraft_ids = reassign_row["Aircraft_IDs"]
                itin_ac = airline_fleets[airline_id][airline_fleets[airline_id]["AircraftID"].isin(itin_aircraft_ids)].copy()
                itin_ac.sort_values(["SizeClass", "Age_years"], ascending=[False, False], inplace=True)
                reassign_ac = itin_ac.iloc[0]  # largest aircraft only

                # ground aircraft
                out_reassign_mask = (
                    (airline_routes[airline_id]["origin"] == reassign_itin_out["origin"])
                    & (airline_routes[airline_id]["destination"] == reassign_itin_out["destination"])
                    & (airline_routes[airline_id]["fuel_stop"] == reassign_itin_out["fuel_stop"])
                )
                in_reassign_mask = (
                    (airline_routes[airline_id]["origin"] == reassign_itin_in["origin"])
                    & (airline_routes[airline_id]["destination"] == reassign_itin_in["destination"])
                    & (airline_routes[airline_id]["fuel_stop"] == reassign_itin_in["fuel_stop"])
                )
                new_origin = -1
                new_destination = -1
                addnl_flights_per_year = 0
                addnl_seat_flights_per_year = 0
                (
                    airlines,
                    airline_routes,
                    airline_fleets,
                    city_data,
                    city_pair_data,
                ) = reassignment.reassign_ac_to_new_route(
                    new_origin,
                    new_destination,
                    out_reassign_mask,
                    in_reassign_mask,
                    reassign_itin_out,
                    reassign_itin_in,
                    reassign_ac,
                    addnl_flights_per_year,
                    addnl_seat_flights_per_year,
                    city_pair_data,
                    city_data,
                    airline_routes,
                    airlines,
                    airline_id,
                    airline_fleets,
                    aircraft_data,
                    demand_coefficients,
                )

                # update masks because an itinerary may have been removed from airline_routes
                out_reassign_mask = (
                    (airline_routes[airline_id]["origin"] == reassign_itin_out["origin"])
                    & (airline_routes[airline_id]["destination"] == reassign_itin_out["destination"])
                    & (airline_routes[airline_id]["fuel_stop"] == reassign_itin_out["fuel_stop"])
                )
                in_reassign_mask = (
                    (airline_routes[airline_id]["origin"] == reassign_itin_in["origin"])
                    & (airline_routes[airline_id]["destination"] == reassign_itin_in["destination"])
                    & (airline_routes[airline_id]["fuel_stop"] == reassign_itin_in["fuel_stop"])
                )
                # update rtn_flt_df
                rtn_flt_df = reassignment.update_profit_tracker(
                    rtn_flt_df,
                    new_origin,
                    new_destination,
                    out_reassign_mask,
                    in_reassign_mask,
                    reassign_itin_out,
                    reassign_ac,
                    airline_routes,
                    airline_fleets,
                    airline_id,
                    city_pair_data,
                    city_data,
                    aircraft_data,
                    FuelCost_USDperGallon,
                    demand_coefficients,
                )

                # if no planes left, remove itinerary from airline_routes and rtn_flt_df
                if len(airline_routes[airline_id].loc[out_reassign_mask, "aircraft_ids"].iloc[0]) == 0:
                    # remove itinerary from airline_routes
                    airline_routes[airline_id] = airline_routes[airline_id].loc[~out_reassign_mask]
                    in_reassign_mask = (
                        (airline_routes[airline_id]["origin"] == reassign_itin_in["origin"])
                        & (airline_routes[airline_id]["destination"] == reassign_itin_in["destination"])
                        & (airline_routes[airline_id]["fuel_stop"] == reassign_itin_in["fuel_stop"])
                    )  # recalculate mask since outbound itinerary has been removed
                    airline_routes[airline_id] = airline_routes[airline_id].loc[~in_reassign_mask]
                    # remove itinerary from rtn_flt_df
                    rtn_flt_df = rtn_flt_df.drop(
                        rtn_flt_df[
                            (rtn_flt_df["Origin"] == reassign_itin_out["origin"]) &
                            (rtn_flt_df["Destination"] == reassign_itin_out["destination"]) &
                            (rtn_flt_df["Fuel_Stop"] == reassign_itin_out["fuel_stop"])
                        ].index
                    )

                # reorder rtn_flt_df
                rtn_flt_df.sort_values("Profit_perSeat", ascending=True, inplace=True)
        else:
            # airline has no loss-making aircraft, so allow airline to reassign previously grounded aircraft
            while (
                len(airlines.loc[airline_id, "Grounded_acft"]) > 0
            ):
                # try to assign the smallest grounded aircraft to a profitable route
                ground_aircraft_ids = airlines.loc[airline_id, "Grounded_acft"]
                ground_ac = airline_fleets[airline_id][airline_fleets[airline_id]["AircraftID"].isin(ground_aircraft_ids)].copy()
                ground_ac.sort_values(["SizeClass", "Age_years"], ascending=[True, True], inplace=True)
                reassign_ac = ground_ac.iloc[0]  # smallest aircraft only

                reassign_new_profit_per_seat = 0.0
                reassign_old_profit_per_seat = 0.0
                (
                    delta_profit_per_seat,
                    new_origin, new_destination,
                    addnl_flights_per_year,
                    addnl_seat_flights_per_year,
                ) = reassignment.find_itin_alternative(
                    city_data,
                    city_pair_data,
                    city_lookup,
                    airline["CountryID"],
                    airline_routes[airline_id],
                    airline_fleets[airline_id],
                    aircraft_data,
                    reassign_ac,
                    demand_coefficients,
                    FuelCost_USDperGallon,
                    reassign_new_profit_per_seat,
                    reassign_old_profit_per_seat,
                )

                # assign the aircraft to the most profitable alternative if not loss-making
                if delta_profit_per_seat > 0.0:
                    out_reassign_mask = None
                    in_reassign_mask = None
                    reassign_itin_out = None
                    reassign_itin_in = None
                    (
                        airlines,
                        airline_routes,
                        airline_fleets,
                        city_data,
                        city_pair_data,
                    ) = reassignment.reassign_ac_to_new_route(
                        new_origin,
                        new_destination,
                        out_reassign_mask,
                        in_reassign_mask,
                        reassign_itin_out,
                        reassign_itin_in,
                        reassign_ac,
                        addnl_flights_per_year,
                        addnl_seat_flights_per_year,
                        city_pair_data,
                        city_data,
                        airline_routes,
                        airlines,
                        airline_id,
                        airline_fleets,
                        aircraft_data,
                        demand_coefficients,
                    )

                else:
                    # no profitable routes available
                    break

            if len(airlines.loc[airline_id, "Grounded_acft"]) > 0:
                # end lease on all remaining grounded aircraft
                for ac_idx in airlines.loc[airline_id, "Grounded_acft"]:
                    # remove aircraft from airline["n_Aircraft"]
                    sizeclass = int(airline_fleets[airline_id].loc[
                        airline_fleets[airline_id]["AircraftID"] == ac_idx, "SizeClass"
                    ].iloc[0])
                    airlines.loc[airline_id, "n_Aircraft"][sizeclass] -= 1
                    
                    # remove aircraft from airline_fleets
                    airline_fleets[airline_id] = airline_fleets[airline_id][
                        airline_fleets[airline_id]["AircraftID"] != ac_idx
                    ]
                airlines.at[airline_id, "Grounded_acft"] = []
            else:
                # allow airline to lease new aircraft, starting with longest range aircraft type first
                aircraft_data.sort_values(by="TypicalRange_m", inplace=True, ascending=False)

                n_aircraft_sum = sum(airlines.loc[airline_id, "n_Aircraft"])
                max_expansion = min(MAX_EXPANSION_PLANES, math.floor(n_aircraft_sum * MAX_EXPANSION_PROPORTION))  # max no. a/c or % of fleet size, whichever is smaller
                n_new_aircraft = 0

                for aircraft_size, aircraft in aircraft_data.iterrows():
                    if n_new_aircraft >= max_expansion:
                        break
                    finished = False
                    while not finished:
                        if n_new_aircraft >= max_expansion:
                            break
                        # create new aircraft, add to airline_fleets and airline's grounded aircraft
                        if len(airline_fleets[airline_id]) == 0:
                            new_ac_id = 0
                        else:
                            new_ac_id = airline_fleets[airline_id]["AircraftID"].max() + 1
                        new_ac_dict = {
                            "AircraftID": int(new_ac_id),
                            "SizeClass": int(aircraft_size),
                            "Age_years": 0,
                            "Lease_USDperMonth": aircraft["LeaseRateNew_USDPerMonth"],
                            "BreguetFactor": (aircraft["Breguet_gradient"] * year) + aircraft["Breguet_intercept"],
                            "RouteOrigin": -1,
                            "RouteDestination": -1,
                            "FuelStop": -1,
                            "Flights_perYear": 0,
                        }
                        new_ac_df = pd.DataFrame(new_ac_dict, index=[0])
                        airline_fleets[airline_id] = pd.concat([airline_fleets[airline_id], new_ac_df], ignore_index=True)

                        reassign_ac = airline_fleets[airline_id].iloc[-1]

                        airlines.at[airline_id, "Grounded_acft"].append(reassign_ac["AircraftID"])

                        reassign_new_profit_per_seat = 0.0
                        reassign_old_profit_per_seat = 0.0
                        (
                            delta_profit_per_seat,
                            new_origin, new_destination,
                            addnl_flights_per_year,
                            addnl_seat_flights_per_year,
                        ) = reassignment.find_itin_alternative(
                            city_data,
                            city_pair_data,
                            city_lookup,
                            airline["CountryID"],
                            airline_routes[airline_id],
                            airline_fleets[airline_id],
                            aircraft_data,
                            reassign_ac,
                            demand_coefficients,
                            FuelCost_USDperGallon,
                            reassign_new_profit_per_seat,
                            reassign_old_profit_per_seat,
                        )

                        # assign the aircraft to the most profitable alternative if not loss-making
                        if delta_profit_per_seat > 0.0:
                            out_reassign_mask = None
                            in_reassign_mask = None
                            reassign_itin_out = None
                            reassign_itin_in = None
                            (
                                airlines,
                                airline_routes,
                                airline_fleets,
                                city_data,
                                city_pair_data,
                            ) = reassignment.reassign_ac_to_new_route(
                                new_origin,
                                new_destination,
                                out_reassign_mask,
                                in_reassign_mask,
                                reassign_itin_out,
                                reassign_itin_in,
                                reassign_ac,
                                addnl_flights_per_year,
                                addnl_seat_flights_per_year,
                                city_pair_data,
                                city_data,
                                airline_routes,
                                airlines,
                                airline_id,
                                airline_fleets,
                                aircraft_data,
                                demand_coefficients,
                            )

                            # add aircraft to airline["n_Aircraft"]
                            airlines.loc[airline_id, "n_Aircraft"][aircraft_size] += 1

                            n_new_aircraft += 1
                        else:
                            # no profitable routes available
                            finished = True
                            # remove aircraft that was being experimented with
                            airline_fleets[airline_id] = airline_fleets[airline_id][
                                airline_fleets[airline_id]["AircraftID"] != reassign_ac["AircraftID"]
                            ]
                            airlines.at[airline_id, "Grounded_acft"].remove(reassign_ac["AircraftID"])

    return airline_routes, airline_fleets, city_pair_data, city_data


def update_itinerary_times(
    airlines: pd.DataFrame,
    airline_routes: list[pd.DataFrame],
    city_data: pd.DataFrame,
    city_pair_data: pd.DataFrame,
    aircraft_data: pd.DataFrame,
    airline_fleets: list[pd.DataFrame],
) -> list[pd.DataFrame]:
    for airline_id, _ in airlines.iterrows():
        for idx, itin in airline_routes[airline_id].iterrows():
            # remove itinerary if no aircraft assigned and warn because this should have been dealt with elsewhere
            if itin["aircraft_ids"] == []:
                airline_routes[airline_id].drop(idx, inplace=True)
                print(f"        Warning: itinerary removed between years due to having no aircraft assigned.")
                continue
            airline_routes[airline_id].loc[idx, "itin_time_hrs"] = acft.calc_itin_time(
                airline_routes[airline_id].loc[idx],
                city_data,
                city_pair_data,
                aircraft_data,
                airline_fleets[airline_id],
            )
    return airline_routes
