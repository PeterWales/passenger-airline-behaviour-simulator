import pandas as pd
import numpy as np
import route
import aircraft as acft
import demand
from scipy.optimize import minimize_scalar
import os
import copy


def initialise_airlines(
    fleet_data: pd.DataFrame,
    country_data: pd.DataFrame,
    run_parameters: pd.DataFrame,
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

    Returns
    -------
    airlines : pd.DataFrame
    """
    census_regions = {
        "Americas": [10, 11, 12],
        "Europe": [13],
        "MiddleEast": [14],
        "Africa": [15],
        "AsiaPacific": [16],
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
                x[1]["GDP"],
                x[1]["Population_2015"],
            ),  # Use population as tiebreaker
            reverse=True,
        )

        # assign aircraft to countries
        region_aircraft = fleet_data[f"Census{region}"].to_list()
        unallocated = region_aircraft.copy()
        for (country_idx, country,) in sorted_region:
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
            for country_airline_idx in range(run_parameters["AirlinesPerCountry"]):
                n_aircraft = [0] * n_aircraft_types
                for aircraft_type in range(n_aircraft_types):
                    base = (
                        country_aircraft[aircraft_type]
                        // run_parameters["AirlinesPerCountry"]
                    )
                    remainder = (
                        country_aircraft[aircraft_type]
                        % run_parameters["AirlinesPerCountry"]
                    )
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

    min_load_factor = 0.8  # minimum load factor for an aircraft to be assigned to a route

    n_cities = len(city_data)
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
    for _, airline in airlines.iterrows():
        airline_id = airline["Airline_ID"]

        # calculate total base RPKs for all routes the airline can operate (assume airlines can only run routes to/from their home country)
        possible_RPKs = 0.0
        distances = []
        for origin_id in city_lookup[airline["CountryID"]]:
            for destination_id in range(n_cities):
                # check outbound and inbound routes exist in city_pair_data
                if (not city_pair_data[
                    (city_pair_data["OriginCityID"] == origin_id)
                    & (city_pair_data["DestinationCityID"] == destination_id)
                ].empty) and (not city_pair_data[
                    (city_pair_data["OriginCityID"] == destination_id)
                    & (city_pair_data["DestinationCityID"] == origin_id)
                ].empty):
                    if (origin_id == destination_id):
                        continue
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

            for __, aircraft in aircraft_data.iterrows():
                if seats <= (smallest_ac * min_load_factor):
                    break
                aircraft_size = aircraft.name
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
                                (seats / aircraft["Seats"] > min_load_factor)
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
                                    demand_coefficients,
                                    capacity_flag_list,
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
                                )
                            )

                            if not (fuel_stop == -1):  # -1 if route not possible with a stop for that aircraft
                                while(
                                    (seats / aircraft["Seats"] > min_load_factor)
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
                                        demand_coefficients,
                                        capacity_flag_list,
                                    )

                        else:
                            break
        
        fleet_df = pd.DataFrame()

        fleet_df["AircraftID"] = aircraft_id_list
        fleet_df["SizeClass"] = aircraft_size_list
        fleet_df["Age_years"] = aircraft_age_list
        fleet_df["Lease_USDperMonth"] = current_lease_list
        fleet_df["BreguetFactor"] = breguet_factor_list
        fleet_df["RouteOrigin"] = origin_id_list
        fleet_df["RouteDestination"] = destination_id_list
        fleet_df["FuelStop"] = fuel_stop_list
        fleet_df["Flights_perYear"] = flights_per_year_list

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
    demand_coefficients: dict[str, float],
    capacity_flag_list: list,
):
    op_hrs_per_year = 6205.0  # =17*365 (assume airports are closed between 11pm and 6am)
    
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

    if fuel_stop == -1:
        fuel_stop_series = None
    else:
        fuel_stop_series = city_data.loc[fuel_stop]

    flights_per_year_list.append(
        acft.calc_flights_per_year(
            city_data.loc[origin_id],
            origin_id,
            city_data.loc[destination_id],
            destination_id,
            aircraft,
            city_pair_data,
            fuel_stop_series,
            fuel_stop
        )
    )

    city_data.loc[origin_id, "Movts_perHr"] += 2.0 * float(flights_per_year_list[-1]) / op_hrs_per_year  # 2* since each flight is return
    city_data.loc[destination_id, "Movts_perHr"] += 2.0 * float(flights_per_year_list[-1]) / op_hrs_per_year

    # flag city if capacity limit exceeded
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

    seat_flights_per_year = flights_per_year_list[-1] * aircraft["Seats"]
    flight_time_hrs = (
        outbound_route["Great_Circle_Distance_m"] / (aircraft["CruiseV_ms"] * 3600)
    )  # could make this more accurate by taking average of different aircraft rather than just the last one
    if not_route_exists:
        outbound_flights_per_year = flights_per_year_list[-1]
    else:
        outbound_mask = (
            (airline_routes[airline_id]["origin"] == origin_id) & 
            (airline_routes[airline_id]["destination"] == destination_id)
        )
        inbound_mask = (
            (airline_routes[airline_id]["origin"] == destination_id) & 
            (airline_routes[airline_id]["destination"] == origin_id)
        )
        outbound_flights_per_year = airline_routes[airline_id].loc[outbound_mask, "flights_per_year"].iloc[0] + flights_per_year_list[-1]
    inbound_flights_per_year = outbound_flights_per_year

    outbound_exp_utility = demand.calc_exp_utility(
        demand_coefficients,
        outbound_route["Fare_Est"],
        flight_time_hrs
        + city_data.loc[origin_id, "Taxi_Out_mins"]
        + city_data.loc[destination_id, "Taxi_In_mins"],
        outbound_flights_per_year,
        fuel_stop,
    )
    inbound_exp_utility = demand.calc_exp_utility(
        demand_coefficients,
        inbound_route["Fare_Est"],
        flight_time_hrs
        + city_data.loc[destination_id, "Taxi_Out_mins"]
        + city_data.loc[origin_id, "Taxi_In_mins"],
        inbound_flights_per_year,
        fuel_stop,
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
    city_pair_data.loc[
        (city_pair_data["OriginCityID"] == origin_id)
        & (city_pair_data["DestinationCityID"] == destination_id),
        "Exp_Utility_Sum"
    ] += outbound_exp_utility
    city_pair_data.loc[
        (city_pair_data["OriginCityID"] == destination_id)
        & (city_pair_data["DestinationCityID"] == origin_id),
        "Exp_Utility_Sum"
    ] += inbound_exp_utility

    # update airline-specific route dataframe
    if not_route_exists:
        airline_routes_newrow_1 = {
            "origin": [origin_id],
            "destination": [destination_id],
            "fare": [outbound_route["Fare_Est"]],
            "aircraft_ids": [[aircraft_id]],
            "flights_per_year": [flights_per_year_list[-1]],
            "seat_flights_per_year": [seat_flights_per_year],
            "exp_utility": [outbound_exp_utility],
            "fuel_stop": [fuel_stop]
        }
        new_df_1 = pd.DataFrame(airline_routes_newrow_1)
        airline_routes_newrow_2 = {
            "origin": [destination_id],
            "destination": [origin_id],
            "fare": [inbound_route["Fare_Est"]],
            "aircraft_ids": [[aircraft_id]],
            "flights_per_year": [flights_per_year_list[-1]],
            "seat_flights_per_year": [seat_flights_per_year],
            "exp_utility": [inbound_exp_utility],
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

        airline_routes[airline_id].loc[outbound_mask, "flights_per_year"] += flights_per_year_list[-1]
        airline_routes[airline_id].loc[inbound_mask, "flights_per_year"] += flights_per_year_list[-1]

        airline_routes[airline_id].loc[outbound_mask, "seat_flights_per_year"] += seat_flights_per_year
        airline_routes[airline_id].loc[inbound_mask, "seat_flights_per_year"] += seat_flights_per_year

        airline_routes[airline_id].loc[outbound_mask, "exp_utility"] = outbound_exp_utility
        airline_routes[airline_id].loc[inbound_mask, "exp_utility"] = inbound_exp_utility

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


def optimise_fares(
    airlines: pd.DataFrame,
    airline_routes: list[pd.DataFrame],
    airline_fleets: list[pd.DataFrame],
    city_pair_data: pd.DataFrame,
    city_data: pd.DataFrame,
    aircraft_data: pd.DataFrame,
    max_fare: float,
    maxiters: int,
    demand_tolerance: float,
    save_folder_path: str
) -> tuple[
    list[pd.DataFrame],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame
]:
    """
    Optimise fares for all airlines, assuming no change in flight schedules. Iterate to find an equilibrium for the base year.

    Parameters
    ----------
    airlines : pd.DataFrame
    airline_routes : list[pd.DataFrame]
    airline_fleets : list[pd.DataFrame]
    city_pair_data : pd.DataFrame
    city_data : pd.DataFrame
    aircraft_data : pd.DataFrame
    maxiters : int
    demand_tolerance : float
    save_folder_path : str

    Returns
    -------
    airline_routes : list[pd.DataFrame]
    city_pair_data : pd.DataFrame
    fare_iters : pd.DataFrame
    demand_iters : pd.DataFrame
    """
    # initialise dataframes tracking convergence
    fare_iters = pd.DataFrame()
    fare_iters["Origin"] = city_pair_data["OriginCityID"]
    fare_iters["Destination"] = city_pair_data["DestinationCityID"]
    fare_iters["Origin_Latitude"] = city_pair_data["Origin_Latitude"]
    fare_iters["Origin_Longitude"] = city_pair_data["Origin_Longitude"]
    fare_iters["Destination_Latitude"] = city_pair_data["Destination_Latitude"]
    fare_iters["Destination_Longitude"] = city_pair_data["Destination_Longitude"]
    fare_iters["base"] = city_pair_data["Mean_Fare_USD"]

    demand_iters = pd.DataFrame()
    demand_iters["Origin"] = city_pair_data["OriginCityID"]
    demand_iters["Destination"] = city_pair_data["DestinationCityID"]
    demand_iters["Origin_Latitude"] = city_pair_data["Origin_Latitude"]
    demand_iters["Origin_Longitude"] = city_pair_data["Origin_Longitude"]
    demand_iters["Destination_Latitude"] = city_pair_data["Destination_Latitude"]
    demand_iters["Destination_Longitude"] = city_pair_data["Destination_Longitude"]
    demand_iters["base"] = city_pair_data["Total_Demand"]

    for iteration in range(maxiters):
        print(f"        Iteration {iteration+1} of fare optimisation")
        # allow each airline to adjust their prices without knowledge of other airlines' choices
        for _, airline in airlines.iterrows():
            for idx, itin in airline_routes[airline["Airline_ID"]].iterrows():
                city_pair = city_pair_data[
                    (city_pair_data["OriginCityID"] == itin["origin"])
                    & (city_pair_data["DestinationCityID"] == itin["destination"])
                ].iloc[0]
                old_fare = itin["fare"]
                new_fare = maximise_itin_profit(
                    itin,
                    airline_fleets[airline["Airline_ID"]],
                    city_pair_data,
                    city_data,
                    aircraft_data,
                    max_fare
                )
                airline_routes[airline["Airline_ID"]].loc[idx, "fare"] = new_fare

                # update city_pair_data with new mean fare
                # can update in place because these fare values are only used for overall O-D demand calculation
                itin_fare_diff = new_fare - old_fare
                prev_mean_fare = city_pair["Mean_Fare_USD"]
                city_pair_data.loc[
                    (city_pair_data["OriginCityID"] == itin["origin"])
                    & (city_pair_data["DestinationCityID"] == itin["destination"]),
                    "Mean_Fare_USD"
                ] = (
                    (prev_mean_fare * city_pair["Seat_Flights_perYear"])
                    + (itin_fare_diff * itin["seat_flights_per_year"])
                ) / city_pair["Seat_Flights_perYear"]

        # update demand for all O-D pairs
        for idx, city_pair in city_pair_data.iterrows():
            city_pair_data.loc[idx, "Total_Demand"] = demand.update_od_demand(city_pair)

        # write new columns to dataframes
        fare_iters[f"iter{iteration}"] = city_pair_data["Mean_Fare_USD"]
        demand_iters[f"iter{iteration}"] = city_pair_data["Total_Demand"]

        # write dataframes to files
        fare_iters.to_csv(os.path.join(save_folder_path, "fare_iters.csv"), index=False)
        demand_iters.to_csv(os.path.join(save_folder_path, "demand_iters.csv"), index=False)

        # check convergence
        if iteration > 0:
            diff = abs(demand_iters[f"iter{iteration}"] - demand_iters[f"iter{iteration-1}"])
            max_diff = diff.max()
            mean_diff = diff.mean()
            print(f"        Maximum demand shift: {max_diff}")
            print(f"        Mean demand shift: {mean_diff}")
            
            if max_diff < demand_tolerance:
                print(f"        Converged after {iteration} iterations")
                break

    return airline_routes, city_pair_data, fare_iters, demand_iters


def maximise_itin_profit(
    airline_route: pd.Series,
    fleet_df: pd.DataFrame,
    city_pair_data: pd.DataFrame,
    city_data: pd.DataFrame,
    aircraft_data: pd.DataFrame,
    max_fare: float
) -> float:
    """
    Maximise profit for a given itinerary by adjusting fare
    """
    fare_bounds = (0, max_fare)

    origin = city_data.loc[airline_route["origin"]]
    destination = city_data.loc[airline_route["destination"]]
    city_pair = city_pair_data[
        (city_pair_data["OriginCityID"] == airline_route["origin"])
        & (city_pair_data["DestinationCityID"] == airline_route["destination"])
    ].iloc[0]

    # Minimise negative profit
    def objective(fare):
        return -itin_profit(
            fare,
            airline_route,
            city_pair,
            origin,
            destination,
            fleet_df,
            aircraft_data,
        )

    result = minimize_scalar(objective, bounds=fare_bounds, method="bounded")
    optimal_fare = result.x
    return optimal_fare


def itin_profit(
    fare: float,  # new airline-specific itinerary fare to test
    airline_route: pd.Series,
    city_pair: pd.Series,
    origin: pd.Series,
    destination: pd.Series,
    fleet_data: pd.DataFrame,
    aircraft_data: pd.DataFrame,
    add_city_pair_seat_flights: int | None = None,
    add_city_pair_exp_utility: float | None = None,
    add_itin_seat_flights: int | None = None,
    add_itin_exp_utility: float | None = None
) -> float:
    """
    Calculate profit for a given itinerary (one-way) and fare
    Note since this function acts on rows of DataFrames, it does not alter the original DataFrames
    """
    # TODO: implement variation in fuel price over time

    # Make a copy of airline_route to surpress warnings
    airline_route = copy.deepcopy(airline_route)

    if add_city_pair_seat_flights is not None:
        city_pair["Seat_Flights_perYear"] += add_city_pair_seat_flights
    if add_city_pair_exp_utility is not None:
        city_pair["Exp_Utility_Sum"] += add_city_pair_exp_utility
    if add_itin_seat_flights is not None:
        airline_route["seat_flights_per_year"] += add_itin_seat_flights
    if add_itin_exp_utility is not None:
        airline_route["exp_utility"] += add_itin_exp_utility

    FuelCost_USDperGallon = 1.5  # PLACEHOLDER

    # update city_pair mean fare (weighted by seats available not seats filled)
    total_revenue = city_pair["Mean_Fare_USD"] * city_pair["Seat_Flights_perYear"]
    total_revenue += ((fare - airline_route["fare"]) * airline_route["seat_flights_per_year"])
    city_pair["Mean_Fare_USD"] = total_revenue / city_pair["Seat_Flights_perYear"]
    city_pair["Total_Demand"] = demand.update_od_demand(city_pair)

    # update airline route fare
    airline_route["fare"] = fare

    # can't sell more tickets than the airline has scheduled
    annual_itin_demand = demand.update_itinerary_demand(city_pair, airline_route)
    tickets_sold = min([annual_itin_demand, airline_route["seat_flights_per_year"]])
    
    annual_itin_revenue = tickets_sold * airline_route["fare"]

    annual_itin_cost = acft.calc_flight_cost(
        airline_route,
        fleet_data,
        aircraft_data,
        city_pair,
        origin,
        destination,
        annual_itin_demand,
        FuelCost_USDperGallon,
    )  # note depends on number of seats sold due to weight
    annual_itin_profit = annual_itin_revenue - annual_itin_cost
    return annual_itin_profit


def reassign_ac_for_profit(
    airlines: pd.DataFrame,
    airline_routes: list[pd.DataFrame],
    airline_fleets: list[pd.DataFrame],
    city_pair_data: pd.DataFrame,
    city_data: pd.DataFrame,
    city_lookup: list,
    aircraft_data: pd.DataFrame,
    demand_coefficients: dict[str, float]
) -> tuple[
    list[pd.DataFrame],
    list[pd.DataFrame],
    pd.DataFrame,
    pd.DataFrame
]:
    """
    Reassign aircraft to different routes to maximise profit per seat
    Airline by airline, starts with the least profitable route and reassigns aircraft to the most profitable alternative route
    Stops when the least profitable aircraft can't be reassigned to a more profitable route

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
    #       - consider grounding aircraft with negative profit and testing grounded aircraft on alternative routes

    op_hrs_per_year = 6205.0  # airport op hours per year = 17*365 (assume airports are closed between 11pm and 6am)

    # iterate over all airlines
    for _, airline in airlines.iterrows():
        print(f"    Reassigning aircraft for airline {airline['Airline_ID']}")

        airline_id = airline["Airline_ID"]

        already_considered = np.full(len(airline_routes[airline_id]), False)
        profit_per_seat_list = []
        origin_list = []
        destination_list = []
        fuel_stop_list = []
        aircraft_id_list = []
        # iterate over all routes operated by that airline
        for out_itin_idx, out_itin in airline_routes[airline_id].iterrows():
            # consider outbound and inbound itineraries together
            if already_considered[out_itin_idx]:
                continue

            in_itin_idx = airline_routes[airline_id][
                (airline_routes[airline_id]["origin"] == out_itin["destination"])
                & (airline_routes[airline_id]["destination"] == out_itin["origin"])
                & (airline_routes[airline_id]["fuel_stop"] == out_itin["fuel_stop"])
            ].index[0]
            in_itin = airline_routes[airline_id].iloc[in_itin_idx]

            city_pair_out = city_pair_data[
                (city_pair_data["OriginCityID"] == out_itin["origin"])
                & (city_pair_data["DestinationCityID"] == out_itin["destination"])
            ].iloc[0]
            city_pair_in = city_pair_data[
                (city_pair_data["OriginCityID"] == out_itin["destination"])
                & (city_pair_data["DestinationCityID"] == out_itin["origin"])
            ].iloc[0]

            origin_list.append(out_itin["origin"])
            destination_list.append(out_itin["destination"])
            fuel_stop_list.append(out_itin["fuel_stop"])
            aircraft_id_list.append(copy.deepcopy(out_itin["aircraft_ids"]))

            # calculate sum of all seats that the airline has assigned to this itinerary
            itin_seats = 0
            for ac_id in out_itin["aircraft_ids"]:
                itin_seats += aircraft_data.loc[airline_fleets[airline_id].loc[ac_id, "SizeClass"], "Seats"]

            # calculate profit per seat of itinerary (outbound + inbound)
            profit_per_seat_list.append(
                (
                    itin_profit(
                        out_itin["fare"],
                        out_itin,
                        city_pair_out,
                        city_data.loc[out_itin["origin"]],
                        city_data.loc[out_itin["destination"]],
                        airline_fleets[airline_id],
                        aircraft_data
                    ) + itin_profit(
                        in_itin["fare"],
                        in_itin,
                        city_pair_in,
                        city_data.loc[out_itin["destination"]],
                        city_data.loc[out_itin["origin"]],
                        airline_fleets[airline_id],
                        aircraft_data
                    )
                ) / itin_seats
            )
            # flag inbound route as already considered
            already_considered[in_itin_idx] = True

        # create dataframe
        rtn_flt_dict = {
            "Origin": origin_list,
            "Destination": destination_list,
            "Fuel_Stop": fuel_stop_list,
            "Profit_perSeat": profit_per_seat_list,
            "Aircraft_IDs": aircraft_id_list
        }
        rtn_flt_df = pd.DataFrame(rtn_flt_dict)

        # reassign aircraft one-by-one to most profitable route and stop when no more profitable routes are available
        finished = False
        while not finished:
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
            fleet_df = airline_fleets[airline_id]
            itin_ac = fleet_df[fleet_df["AircraftID"].isin(itin_aircraft_ids)]
            itin_ac.sort_values(["SizeClass", "Age_years"], ascending=[False, False], inplace=True)
            reassign_ac = itin_ac.iloc[0]  # largest aircraft only

            # find new profit per seat of route that aircraft is being removed from
            if len(itin_ac) == 1:
                # no aircraft left on this route after reallocation
                reassign_new_profit_per_seat = 0.0
            else:
                seat_flights_per_year = reassign_ac["Flights_perYear"] * aircraft_data.loc[reassign_ac["SizeClass"], "Seats"]
                reassign_itin_out["flights_per_year"] -= reassign_ac["Flights_perYear"]
                reassign_itin_out["seat_flights_per_year"] -= seat_flights_per_year
                reassign_itin_out["aircraft_ids"].remove(reassign_ac["AircraftID"])
                flight_time = (
                    city_pair_data[
                        (city_pair_data["OriginCityID"] == reassign_itin_out["origin"])
                        & (city_pair_data["DestinationCityID"] == reassign_itin_out["destination"])
                    ].iloc[0]["Great_Circle_Distance_m"]
                    / (aircraft_data.loc[reassign_ac["SizeClass"], "CruiseV_ms"] * 3600)
                )
                itin_time_out = (
                    flight_time
                    + city_data.loc[reassign_itin_out["origin"], "Taxi_Out_mins"]
                    + city_data.loc[reassign_itin_out["destination"], "Taxi_In_mins"]
                )
                old_out_exp_utility = reassign_itin_out["exp_utility"]
                reassign_itin_out["exp_utility"] = demand.calc_exp_utility(
                    demand_coefficients,
                    reassign_itin_out["fare"],
                    itin_time_out,
                    reassign_itin_out["flights_per_year"],
                    reassign_itin_out["fuel_stop"],
                )
                old_in_exp_utility = reassign_itin_in["exp_utility"]
                reassign_itin_in["flights_per_year"] -= reassign_ac["Flights_perYear"]
                reassign_itin_in["seat_flights_per_year"] -= seat_flights_per_year
                reassign_itin_in["aircraft_ids"].remove(reassign_ac["AircraftID"])
                itin_time_in = (
                    flight_time
                    + city_data.loc[reassign_itin_in["destination"], "Taxi_Out_mins"]
                    + city_data.loc[reassign_itin_in["origin"], "Taxi_In_mins"]
                )
                reassign_itin_in["exp_utility"] = demand.calc_exp_utility(
                    demand_coefficients,
                    reassign_itin_in["fare"],
                    itin_time_in,
                    reassign_itin_in["flights_per_year"],
                    reassign_itin_in["fuel_stop"],
                )
                origin = city_data.loc[reassign_itin_out["origin"]]
                destination = city_data.loc[reassign_itin_out["destination"]]
                city_pair = city_pair_data[
                    (city_pair_data["OriginCityID"] == reassign_itin_out["origin"])
                    & (city_pair_data["DestinationCityID"] == reassign_itin_out["destination"])
                ].iloc[0]

                # calculate sum of all seats that the airline has assigned to this itinerary, minus the aircraft being reassigned
                itin_seats = 0
                for ac_id in reassign_itin_out["aircraft_ids"]:
                    itin_seats += aircraft_data.loc[airline_fleets[airline_id].loc[ac_id, "SizeClass"], "Seats"]

                reassign_new_profit_per_seat = (
                    itin_profit(
                        reassign_itin_out["fare"],
                        reassign_itin_out,
                        city_pair,
                        origin,
                        destination,
                        airline_fleets[airline_id],
                        aircraft_data,
                        add_city_pair_seat_flights = -seat_flights_per_year,
                        add_city_pair_exp_utility = reassign_itin_out["exp_utility"] - reassign_itin_out["exp_utility"],
                    ) + itin_profit(
                        reassign_itin_in["fare"],
                        reassign_itin_in,
                        city_pair,
                        destination,
                        origin,
                        airline_fleets[airline_id],
                        aircraft_data,
                        add_city_pair_seat_flights=-seat_flights_per_year,
                        add_city_pair_exp_utility=reassign_itin_in["exp_utility"] - reassign_itin_in["exp_utility"],
                    )
                ) / itin_seats

                # reset reassign_itin_out and reassign_itin_in to avoid mutability issues
                reassign_itin_out["flights_per_year"] += reassign_ac["Flights_perYear"]
                reassign_itin_out["seat_flights_per_year"] += seat_flights_per_year
                reassign_itin_out["aircraft_ids"].append(int(reassign_ac["AircraftID"]))
                reassign_itin_out["exp_utility"] = old_out_exp_utility
                reassign_itin_in["flights_per_year"] += reassign_ac["Flights_perYear"]
                reassign_itin_in["seat_flights_per_year"] += seat_flights_per_year
                reassign_itin_in["aircraft_ids"].append(int(reassign_ac["AircraftID"]))
                reassign_itin_in["exp_utility"] = old_in_exp_utility

            # test new itineraries one-by-one and save the resulting change in profit per seat
            delta_profit_per_seat = 0.0
            new_origin = -1
            new_destination = -1
            for __, city_pair in city_pair_data.iterrows():
                # check whether origin or destination are in country where airline is located
                airline_country = airlines.loc[airline_id, "CountryID"]
                if (
                    city_pair["OriginCityID"] in city_lookup[airline_country]
                    or city_pair["DestinationCityID"] in city_lookup[airline_country]
                ):
                    # check aircraft has enough range and runways are long enough
                    if (
                        city_pair["Great_Circle_Distance_m"] < aircraft_data.loc[reassign_ac["SizeClass"], "TypicalRange_m"]
                        and city_data.loc[city_pair["OriginCityID"], "LongestRunway_m"] > aircraft_data.loc[reassign_ac["SizeClass"], "TakeoffDist_m"]
                        and city_data.loc[city_pair["OriginCityID"], "LongestRunway_m"] > aircraft_data.loc[reassign_ac["SizeClass"], "LandingDist_m"]
                        and city_data.loc[city_pair["DestinationCityID"], "LongestRunway_m"] > aircraft_data.loc[reassign_ac["SizeClass"], "TakeoffDist_m"]
                        and city_data.loc[city_pair["DestinationCityID"], "LongestRunway_m"] > aircraft_data.loc[reassign_ac["SizeClass"], "LandingDist_m"]
                    ):
                        # check adding aircraft won't exceed either city's capacity
                        origin_id = city_pair["OriginCityID"]
                        destination_id = city_pair["DestinationCityID"]
                        flights_per_year = acft.calc_flights_per_year(
                            city_data.loc[origin_id],
                            origin_id,
                            city_data.loc[destination_id],
                            destination_id,
                            aircraft_data.loc[reassign_ac["SizeClass"]],
                            city_pair_data,
                            None,
                            -1
                        )
                        if (
                            city_data.loc[origin_id, "Movts_perHr"] + (2*float(flights_per_year)/op_hrs_per_year) <= city_data.loc[origin_id, "Capacity_MovtsPerHr"]
                            and city_data.loc[destination_id, "Movts_perHr"] + (2*float(flights_per_year)/op_hrs_per_year) <= city_data.loc[destination_id, "Capacity_MovtsPerHr"]
                        ):
                            # new itinerary is possible
                            flights_per_year = acft.calc_flights_per_year(
                                city_data.loc[origin_id],
                                origin_id,
                                city_data.loc[destination_id],
                                destination_id,
                                aircraft_data.loc[reassign_ac["SizeClass"]],
                                city_pair_data,
                                None,
                                -1
                            )
                            seat_flights_per_year = flights_per_year * aircraft_data.loc[reassign_ac["SizeClass"], "Seats"]

                            # check whether airline already flies this route
                            if not airline_routes[airline_id][
                                (airline_routes[airline_id]["origin"] == origin_id)
                                & (airline_routes[airline_id]["destination"] == destination_id)
                                & (airline_routes[airline_id]["fuel_stop"] == -1)
                            ].empty:
                                # calculate old profit per seat of new route without new aircraft
                                out_itin_mask = (
                                    (airline_routes[airline_id]["origin"] == origin_id)
                                    & (airline_routes[airline_id]["destination"] == destination_id)
                                    & (airline_routes[airline_id]["fuel_stop"] == -1)
                                )
                                in_itin_mask = (
                                    (airline_routes[airline_id]["origin"] == destination_id)
                                    & (airline_routes[airline_id]["destination"] == origin_id)
                                    & (airline_routes[airline_id]["fuel_stop"] == -1)
                                )
                                city_pair_in = city_pair_data[
                                    (city_pair_data["OriginCityID"] == city_pair["DestinationCityID"])
                                    & (city_pair_data["DestinationCityID"] == city_pair["OriginCityID"])
                                ].iloc[0]

                                # calculate sum of all seats that the airline has already assigned to this itinerary
                                itin_seats = 0
                                for ac_id in airline_routes[airline_id].loc[out_itin_mask, "aircraft_ids"].iloc[0]:
                                    itin_seats += aircraft_data.loc[airline_fleets[airline_id].loc[ac_id, "SizeClass"], "Seats"]

                                test_itin_out = airline_routes[airline_id].loc[out_itin_mask]
                                test_itin_in = airline_routes[airline_id].loc[in_itin_mask]

                                new_itin_old_profit_per_seat = (
                                    itin_profit(
                                        test_itin_out["fare"].iloc[0],
                                        test_itin_out.iloc[0],
                                        city_pair,
                                        city_data.loc[origin_id],
                                        city_data.loc[destination_id],
                                        airline_fleets[airline_id],
                                        aircraft_data,
                                    ) + itin_profit(
                                        test_itin_in["fare"].iloc[0],
                                        test_itin_in.iloc[0],
                                        city_pair_in,
                                        city_data.loc[destination_id],
                                        city_data.loc[origin_id],
                                        airline_fleets[airline_id],
                                        aircraft_data,
                                    )
                                ) / itin_seats

                                # calculate new profit per seat after new aircraft assigned
                                test_itin_out["flights_per_year"].iloc[0] += flights_per_year
                                test_itin_out["seat_flights_per_year"].iloc[0] += seat_flights_per_year
                                test_itin_out["aircraft_ids"].iloc[0].append(int(reassign_ac["AircraftID"]))
                                flight_time = (
                                    city_pair_data[
                                        (city_pair_data["OriginCityID"] == origin_id)
                                        & (city_pair_data["DestinationCityID"] == destination_id)
                                    ].iloc[0]["Great_Circle_Distance_m"]
                                    / (aircraft_data.loc[reassign_ac["SizeClass"], "CruiseV_ms"] * 3600)
                                )
                                itin_time_out = (
                                    flight_time
                                    + city_data.loc[origin_id, "Taxi_Out_mins"]
                                    + city_data.loc[destination_id, "Taxi_In_mins"]
                                )
                                out_old_exp_utility = test_itin_out["exp_utility"].iloc[0]
                                test_itin_out["exp_utility"].iloc[0] = demand.calc_exp_utility(
                                    demand_coefficients,
                                    test_itin_out["fare"].iloc[0],
                                    itin_time_out,
                                    test_itin_out["flights_per_year"].iloc[0],
                                    -1
                                )
                                test_itin_in["flights_per_year"].iloc[0] += flights_per_year
                                test_itin_in["seat_flights_per_year"].iloc[0] += seat_flights_per_year
                                test_itin_in["aircraft_ids"].iloc[0].append(int(reassign_ac["AircraftID"]))
                                itin_time_in = (
                                    flight_time
                                    + city_data.loc[destination_id, "Taxi_Out_mins"]
                                    + city_data.loc[origin_id, "Taxi_In_mins"]
                                )
                                in_old_exp_utility = test_itin_in["exp_utility"].iloc[0]
                                test_itin_in["exp_utility"].iloc[0] = demand.calc_exp_utility(
                                    demand_coefficients,
                                    test_itin_in["fare"].iloc[0],
                                    itin_time_in,
                                    test_itin_in["flights_per_year"].iloc[0],
                                    -1
                                )

                                # calculate sum of all seats that the airline has assigned to this itinerary including reassigned aircraft
                                itin_seats += aircraft_data.loc[reassign_ac["SizeClass"], "Seats"]

                                new_itin_new_profit_per_seat = (
                                    itin_profit(
                                        test_itin_out["fare"].iloc[0],
                                        test_itin_out.iloc[0],
                                        city_pair,
                                        city_data.loc[origin_id],
                                        city_data.loc[destination_id],
                                        airline_fleets[airline_id],
                                        aircraft_data,
                                        add_city_pair_seat_flights=seat_flights_per_year,
                                        add_city_pair_exp_utility=(
                                            test_itin_out["exp_utility"].iloc[0]
                                            - airline_routes[airline_id].loc[out_itin_mask, "exp_utility"].iloc[0]
                                        )
                                    ) + itin_profit(
                                        test_itin_in["fare"].iloc[0],
                                        test_itin_in.iloc[0],
                                        city_pair_in,
                                        city_data.loc[destination_id],
                                        city_data.loc[origin_id],
                                        airline_fleets[airline_id],
                                        aircraft_data,
                                        add_city_pair_seat_flights=seat_flights_per_year,
                                        add_city_pair_exp_utility=(
                                            test_itin_in["exp_utility"].iloc[0]
                                            - airline_routes[airline_id].loc[in_itin_mask, "exp_utility"].iloc[0]
                                        )
                                    )
                                ) / itin_seats

                                # reset test_itin_out and test_itin_in to avoid mutability issues
                                test_itin_out["flights_per_year"].iloc[0] -= flights_per_year
                                test_itin_out["seat_flights_per_year"].iloc[0] -= seat_flights_per_year
                                test_itin_out["aircraft_ids"].iloc[0].remove(reassign_ac["AircraftID"])
                                test_itin_out["exp_utility"].iloc[0] = out_old_exp_utility
                                test_itin_in["flights_per_year"].iloc[0] -= flights_per_year
                                test_itin_in["seat_flights_per_year"].iloc[0] -= seat_flights_per_year
                                test_itin_in["aircraft_ids"].iloc[0].remove(reassign_ac["AircraftID"])
                                test_itin_in["exp_utility"].iloc[0] = in_old_exp_utility
                            else:
                                # airline doesn't already fly this route
                                new_itin_old_profit_per_seat = 0.0
                                test_itin_out = {
                                    "origin": origin_id,
                                    "destination": destination_id,
                                    "fare": city_pair["Mean_Fare_USD"],
                                    "aircraft_ids": [reassign_ac["AircraftID"]],
                                    "flights_per_year": flights_per_year,
                                    "seat_flights_per_year": seat_flights_per_year,
                                    "exp_utility": 0,
                                    "fuel_stop": -1
                                }
                                test_itin_in = {
                                    "origin": destination_id,
                                    "destination": origin_id,
                                    "fare": city_pair_in["Mean_Fare_USD"],
                                    "aircraft_ids": [reassign_ac["AircraftID"]],
                                    "flights_per_year": flights_per_year,
                                    "seat_flights_per_year": seat_flights_per_year,
                                    "exp_utility": 0,
                                    "fuel_stop": -1
                                }
                                test_itin_out = pd.Series(test_itin_out)
                                test_itin_in = pd.Series(test_itin_in)
                                test_flight_time = city_pair["Great_Circle_Distance_m"] / (aircraft_data.loc[reassign_ac["SizeClass"], "CruiseV_ms"] * 3600)
                                test_itin_time_out = test_flight_time + city_data.loc[origin_id, "Taxi_Out_mins"] + city_data.loc[destination_id, "Taxi_In_mins"]
                                test_itin_time_in = test_flight_time + city_data.loc[destination_id, "Taxi_Out_mins"] + city_data.loc[origin_id, "Taxi_In_mins"]
                                test_itin_out_exp_utility = demand.calc_exp_utility(
                                    demand_coefficients,
                                    test_itin_out["fare"],
                                    test_itin_time_out,
                                    flights_per_year,
                                    -1
                                )
                                test_itin_in_exp_utility = demand.calc_exp_utility(
                                    demand_coefficients,
                                    test_itin_in["fare"],
                                    test_itin_time_in,
                                    flights_per_year,
                                    -1
                                )
                                test_itin_out["exp_utility"] = test_itin_out_exp_utility
                                test_itin_in["exp_utility"] = test_itin_in_exp_utility
                                new_itin_new_profit_per_seat = (
                                    itin_profit(
                                        test_itin_out["fare"],
                                        test_itin_out,
                                        city_pair,
                                        city_data.loc[origin_id],
                                        city_data.loc[destination_id],
                                        airline_fleets[airline_id],
                                        aircraft_data,
                                        add_city_pair_seat_flights=seat_flights_per_year,
                                        add_city_pair_exp_utility=test_itin_out_exp_utility
                                    ) + itin_profit(
                                        test_itin_in["fare"],
                                        test_itin_in,
                                        city_pair_in,
                                        city_data.loc[destination_id],
                                        city_data.loc[origin_id],
                                        airline_fleets[airline_id],
                                        aircraft_data,
                                        add_city_pair_seat_flights=seat_flights_per_year,
                                        add_city_pair_exp_utility=test_itin_in_exp_utility
                                    )
                                ) / aircraft_data.loc[reassign_ac["SizeClass"], "Seats"]  # seats on this itinerary are only provided by reassigned aircraft

                            # calculate change in profit per seat
                            test_delta_profit_per_seat = (
                                new_itin_new_profit_per_seat - new_itin_old_profit_per_seat
                                + reassign_new_profit_per_seat - reassign_old_profit_per_seat
                            )

                            # save if beneficial
                            if test_delta_profit_per_seat > delta_profit_per_seat:
                                delta_profit_per_seat = test_delta_profit_per_seat
                                new_origin = origin_id
                                new_destination = destination_id
                                addnl_flights_per_year = flights_per_year
                                addnl_seat_flights_per_year = seat_flights_per_year

            # assign the aircraft to the most profitable alternative if beneficial
            if delta_profit_per_seat > 0.0:
                old_city_pair_out = city_pair_data[
                    (city_pair_data["OriginCityID"] == reassign_itin_out["origin"])
                    & (city_pair_data["DestinationCityID"] == reassign_itin_out["destination"])
                ].iloc[0]
                old_city_pair_in = city_pair_data[
                    (city_pair_data["OriginCityID"] == reassign_itin_out["destination"])
                    & (city_pair_data["DestinationCityID"] == reassign_itin_out["origin"])
                ].iloc[0]
                new_city_pair_out = city_pair_data[
                    (city_pair_data["OriginCityID"] == new_origin)
                    & (city_pair_data["DestinationCityID"] == new_destination)
                ].iloc[0]
                new_city_pair_in = city_pair_data[
                    (city_pair_data["OriginCityID"] == new_destination)
                    & (city_pair_data["DestinationCityID"] == new_origin)
                ].iloc[0]

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
                
                new_itin_flight_time = new_city_pair_out["Great_Circle_Distance_m"] / (aircraft_data.loc[reassign_ac["SizeClass"], "CruiseV_ms"] * 3600)
                new_itin_time_out = new_itin_flight_time + city_data.loc[new_origin, "Taxi_Out_mins"] + city_data.loc[new_destination, "Taxi_In_mins"]
                new_itin_time_in = new_itin_flight_time + city_data.loc[new_destination, "Taxi_Out_mins"] + city_data.loc[new_origin, "Taxi_In_mins"]

                # save old exp(utility) for calculating deltas for city_pair_data
                reassign_old_utility_out = reassign_itin_out["exp_utility"]
                reassign_old_utility_in = reassign_itin_in["exp_utility"]

                # adjust city_pair_data Seat_Flights_perYear
                city_pair_data.loc[
                    (city_pair_data["OriginCityID"] == reassign_itin_out["origin"])
                    & (city_pair_data["DestinationCityID"] == reassign_itin_out["destination"]),
                    "Seat_Flights_perYear"
                ] -= reassign_ac["Flights_perYear"] * aircraft_data.loc[reassign_ac["SizeClass"], "Seats"]
                city_pair_data.loc[
                    (city_pair_data["OriginCityID"] == reassign_itin_out["destination"])
                    & (city_pair_data["DestinationCityID"] == reassign_itin_out["origin"]),
                    "Seat_Flights_perYear"
                ] -= reassign_ac["Flights_perYear"] * aircraft_data.loc[reassign_ac["SizeClass"], "Seats"]
                city_pair_data.loc[
                    (city_pair_data["OriginCityID"] == new_origin)
                    & (city_pair_data["DestinationCityID"] == new_destination),
                    "Seat_Flights_perYear"
                ] += addnl_seat_flights_per_year
                city_pair_data.loc[
                    (city_pair_data["OriginCityID"] == new_destination)
                    & (city_pair_data["DestinationCityID"] == new_origin),
                    "Seat_Flights_perYear"
                ] += addnl_seat_flights_per_year

                # adjust aircraft data in airline_fleets
                airline_fleets[airline_id].loc[
                    airline_fleets[airline_id]["AircraftID"] == reassign_ac["AircraftID"],
                    "RouteOrigin"
                ] = new_origin
                airline_fleets[airline_id].loc[
                    airline_fleets[airline_id]["AircraftID"] == reassign_ac["AircraftID"],
                    "RouteDestination"
                ] = new_destination
                airline_fleets[airline_id].loc[
                    airline_fleets[airline_id]["AircraftID"] == reassign_ac["AircraftID"],
                    "FuelStop"
                ] = -1
                airline_fleets[airline_id].loc[
                    airline_fleets[airline_id]["AircraftID"] == reassign_ac["AircraftID"],
                    "Flights_perYear"
                ] = addnl_flights_per_year

                # remove aircraft from old airline itinerary
                airline_routes[airline_id].loc[out_reassign_mask, "aircraft_ids"].iloc[0].remove(reassign_ac["AircraftID"])
                airline_routes[airline_id].loc[in_reassign_mask, "aircraft_ids"].iloc[0].remove(reassign_ac["AircraftID"])
                airline_routes[airline_id].loc[out_reassign_mask, "flights_per_year"] -= reassign_ac["Flights_perYear"]
                airline_routes[airline_id].loc[in_reassign_mask, "flights_per_year"] -= reassign_ac["Flights_perYear"]
                airline_routes[airline_id].loc[out_reassign_mask, "seat_flights_per_year"] -= reassign_ac["Flights_perYear"] * aircraft_data.loc[reassign_ac["SizeClass"], "Seats"]
                airline_routes[airline_id].loc[in_reassign_mask, "seat_flights_per_year"] -= reassign_ac["Flights_perYear"] * aircraft_data.loc[reassign_ac["SizeClass"], "Seats"]
                reassign_itin_flight_time = old_city_pair_out["Great_Circle_Distance_m"] / (aircraft_data.loc[reassign_ac["SizeClass"], "CruiseV_ms"] * 3600)
                reassign_itin_time_out = reassign_itin_flight_time + city_data.loc[reassign_itin_out["origin"], "Taxi_Out_mins"] + city_data.loc[reassign_itin_out["destination"], "Taxi_In_mins"]
                reassign_itin_time_in = reassign_itin_flight_time + city_data.loc[reassign_itin_out["destination"], "Taxi_Out_mins"] + city_data.loc[reassign_itin_out["origin"], "Taxi_In_mins"]
                airline_routes[airline_id].loc[out_reassign_mask, "exp_utility"] = demand.calc_exp_utility(
                    demand_coefficients,
                    airline_routes[airline_id].loc[out_reassign_mask, "fare"].iloc[0],
                    reassign_itin_time_out,
                    airline_routes[airline_id].loc[out_reassign_mask, "flights_per_year"].iloc[0],  # already been adjusted
                    -1
                )
                airline_routes[airline_id].loc[in_reassign_mask, "exp_utility"] = demand.calc_exp_utility(
                    demand_coefficients,
                    airline_routes[airline_id].loc[in_reassign_mask, "fare"].iloc[0],
                    reassign_itin_time_in,
                    airline_routes[airline_id].loc[in_reassign_mask, "flights_per_year"].iloc[0],  # already been adjusted
                    -1
                )
                # calculate exp(utility) deltas for city_pair_data
                reassign_delta_exp_utility_out = airline_routes[airline_id].loc[out_reassign_mask, "exp_utility"].iloc[0] - reassign_old_utility_out
                reassign_delta_exp_utility_in = airline_routes[airline_id].loc[in_reassign_mask, "exp_utility"].iloc[0] - reassign_old_utility_in

                # check whether new itinerary already exists
                if not airline_routes[airline_id][
                    (airline_routes[airline_id]["origin"] == new_origin)
                    & (airline_routes[airline_id]["destination"] == new_destination)
                    & (airline_routes[airline_id]["fuel_stop"] == -1)
                ].empty:
                    # airline already flies this route
                    out_itin_mask = (
                        (airline_routes[airline_id]["origin"] == new_origin)
                        & (airline_routes[airline_id]["destination"] == new_destination)
                        & (airline_routes[airline_id]["fuel_stop"] == -1)
                    )
                    in_itin_mask = (
                        (airline_routes[airline_id]["origin"] == new_destination)
                        & (airline_routes[airline_id]["destination"] == new_origin)
                        & (airline_routes[airline_id]["fuel_stop"] == -1)
                    )

                    # save old exp(utility) for calculating deltas for city_pair_data
                    itin_old_utility_out = airline_routes[airline_id].loc[out_itin_mask, "exp_utility"].iloc[0]
                    itin_old_utility_in = airline_routes[airline_id].loc[in_itin_mask, "exp_utility"].iloc[0]

                    # add aircraft to new airline itinerary
                    airline_routes[airline_id].loc[out_itin_mask, "aircraft_ids"].iloc[0].append(int(reassign_ac["AircraftID"]))
                    airline_routes[airline_id].loc[in_itin_mask, "aircraft_ids"].iloc[0].append(int(reassign_ac["AircraftID"]))
                    airline_routes[airline_id].loc[out_itin_mask, "flights_per_year"] += addnl_flights_per_year
                    airline_routes[airline_id].loc[in_itin_mask, "flights_per_year"] += addnl_flights_per_year
                    airline_routes[airline_id].loc[out_itin_mask, "seat_flights_per_year"] += addnl_seat_flights_per_year
                    airline_routes[airline_id].loc[in_itin_mask, "seat_flights_per_year"] += addnl_seat_flights_per_year
                    airline_routes[airline_id].loc[out_itin_mask, "exp_utility"] = demand.calc_exp_utility(
                        demand_coefficients,
                        airline_routes[airline_id].loc[out_itin_mask, "fare"].iloc[0],
                        new_itin_time_out,
                        airline_routes[airline_id].loc[out_itin_mask, "flights_per_year"].iloc[0],
                        -1
                    )
                    airline_routes[airline_id].loc[in_itin_mask, "exp_utility"] = demand.calc_exp_utility(
                        demand_coefficients,
                        airline_routes[airline_id].loc[in_itin_mask, "fare"].iloc[0],
                        new_itin_time_in,
                        airline_routes[airline_id].loc[in_itin_mask, "flights_per_year"].iloc[0],
                        -1
                    )

                    # calculate exp(utility) deltas for city_pair_data
                    itin_delta_exp_utility_out = airline_routes[airline_id].loc[out_itin_mask, "exp_utility"].iloc[0] - itin_old_utility_out
                    itin_delta_exp_utility_in = airline_routes[airline_id].loc[in_itin_mask, "exp_utility"].iloc[0] - itin_old_utility_in
                else:
                    # airline doesn't already fly this route
                    new_itin_out = {
                        "origin": new_origin,
                        "destination": new_destination,
                        "fare": new_city_pair_out["Mean_Fare_USD"],
                        "aircraft_ids": [reassign_ac["AircraftID"]],
                        "flights_per_year": addnl_flights_per_year,
                        "seat_flights_per_year": addnl_seat_flights_per_year,
                        "exp_utility": 0,
                        "fuel_stop": -1
                    }
                    new_itin_in = {
                        "origin": new_destination,
                        "destination": new_origin,
                        "fare": new_city_pair_in["Mean_Fare_USD"],
                        "aircraft_ids": [reassign_ac["AircraftID"]],
                        "flights_per_year": addnl_flights_per_year,
                        "seat_flights_per_year": addnl_seat_flights_per_year,
                        "exp_utility": 0,
                        "fuel_stop": -1
                    }
                    new_itin_out = pd.Series(new_itin_out)
                    new_itin_in = pd.Series(new_itin_in)
                    new_itin_out_exp_utility = demand.calc_exp_utility(
                        demand_coefficients,
                        new_itin_out["fare"],
                        new_itin_time_out,
                        addnl_flights_per_year,
                        -1
                    )
                    new_itin_in_exp_utility = demand.calc_exp_utility(
                        demand_coefficients,
                        new_itin_in["fare"],
                        new_itin_time_in,
                        addnl_flights_per_year,
                        -1
                    )
                    new_itin_out["exp_utility"] = new_itin_out_exp_utility
                    new_itin_in["exp_utility"] = new_itin_in_exp_utility
                    airline_routes[airline_id] = pd.concat(
                        [
                            airline_routes[airline_id],
                            pd.DataFrame([new_itin_out]),
                            pd.DataFrame([new_itin_in])
                        ],
                        ignore_index=True,
                        axis=0  # specify row-wise concatenation
                    )

                    # calculate exp(utility) deltas for city_pair_data (previous utility was zero)
                    itin_delta_exp_utility_out = new_itin_out_exp_utility
                    itin_delta_exp_utility_in = new_itin_in_exp_utility

                    # add itinerary to rtn_flt_df
                    new_itin_dict = {
                        "Origin": new_origin,
                        "Destination": new_destination,
                        "Fuel_Stop": -1,
                        "Profit_perSeat": 0,  # recalculated later
                        "Aircraft_IDs": [reassign_ac["AircraftID"]]
                    }
                    new_itin_df = pd.DataFrame(new_itin_dict)
                    rtn_flt_df = pd.concat([rtn_flt_df, new_itin_df], ignore_index=True)

                # adjust city_data movements
                city_data.loc[reassign_itin_out["origin"], "Movts_perHr"] -= 2.0 * float(reassign_ac["Flights_perYear"])/op_hrs_per_year
                city_data.loc[reassign_itin_out["destination"], "Movts_perHr"] -= 2.0 * float(reassign_ac["Flights_perYear"])/op_hrs_per_year
                city_data.loc[new_origin, "Movts_perHr"] += 2.0 * float(addnl_flights_per_year)/op_hrs_per_year
                city_data.loc[new_destination, "Movts_perHr"] += 2.0 * float(addnl_flights_per_year)/op_hrs_per_year

                # adjust city_pair_data Exp_Utility_Sum
                city_pair_data.loc[
                    (city_pair_data["OriginCityID"] == reassign_itin_out["origin"])
                    & (city_pair_data["DestinationCityID"] == reassign_itin_out["destination"]),
                    "Exp_Utility_Sum"
                ] += reassign_delta_exp_utility_out
                city_pair_data.loc[
                    (city_pair_data["OriginCityID"] == reassign_itin_out["destination"])
                    & (city_pair_data["DestinationCityID"] == reassign_itin_out["origin"]),
                    "Exp_Utility_Sum"
                ] += reassign_delta_exp_utility_in
                city_pair_data.loc[
                    (city_pair_data["OriginCityID"] == new_origin)
                    & (city_pair_data["DestinationCityID"] == new_destination),
                    "Exp_Utility_Sum"
                ] += itin_delta_exp_utility_out
                city_pair_data.loc[
                    (city_pair_data["OriginCityID"] == new_destination)
                    & (city_pair_data["DestinationCityID"] == new_origin),
                    "Exp_Utility_Sum"
                ] += itin_delta_exp_utility_in

                # recalculate profit per seat of altered itineraries
                updated_reassign_itin_out = airline_routes[airline_id].loc[out_reassign_mask]
                updated_reassign_itin_in = airline_routes[airline_id].loc[in_reassign_mask]
                updated_city_pair_out = city_pair_data.loc[
                    (city_pair_data["OriginCityID"] == reassign_itin_out["origin"])
                    & (city_pair_data["DestinationCityID"] == reassign_itin_out["destination"])
                ].iloc[0]
                updated_city_pair_in = city_pair_data.loc[
                    (city_pair_data["OriginCityID"] == reassign_itin_out["destination"])
                    & (city_pair_data["DestinationCityID"] == reassign_itin_out["origin"])
                ].iloc[0]
                # calculate sum of all seats that the airline has assigned to this itinerary minus reassigned aircraft
                itin_seats = 0
                for ac_id in updated_reassign_itin_out["aircraft_ids"].iloc[0]:
                    itin_seats += aircraft_data.loc[airline_fleets[airline_id].loc[ac_id, "SizeClass"], "Seats"]
                if itin_seats == 0:
                    rtn_flt_df.loc[
                        (rtn_flt_df["Origin"] == reassign_itin_out["origin"])
                        & (rtn_flt_df["Destination"] == reassign_itin_out["destination"])
                        & (rtn_flt_df["Fuel_Stop"] == reassign_itin_out["fuel_stop"]),
                        "Profit_perSeat"
                    ] = 0.0
                else:
                    rtn_flt_df.loc[
                        (rtn_flt_df["Origin"] == reassign_itin_out["origin"])
                        & (rtn_flt_df["Destination"] == reassign_itin_out["destination"])
                        & (rtn_flt_df["Fuel_Stop"] == reassign_itin_out["fuel_stop"]),
                        "Profit_perSeat"
                    ] = (
                        itin_profit(
                            updated_reassign_itin_out["fare"].iloc[0],
                            updated_reassign_itin_out.iloc[0],
                            updated_city_pair_out,
                            city_data.loc[reassign_itin_out["origin"]],
                            city_data.loc[reassign_itin_out["destination"]],
                            airline_fleets[airline_id],
                            aircraft_data,
                            # city_pair_data is already updated
                        ) + itin_profit(
                            updated_reassign_itin_in["fare"].iloc[0],
                            updated_reassign_itin_in.iloc[0],
                            updated_city_pair_in,
                            city_data.loc[reassign_itin_out["destination"]],
                            city_data.loc[reassign_itin_out["origin"]],
                            airline_fleets[airline_id],
                            aircraft_data,
                            # city_pair_data is already updated
                        )
                    ) / itin_seats

                updated_new_itin_out = airline_routes[airline_id].loc[
                    (airline_routes[airline_id]["origin"] == new_origin)
                    & (airline_routes[airline_id]["destination"] == new_destination)
                    & (airline_routes[airline_id]["fuel_stop"] == -1)
                ]
                updated_new_itin_in = airline_routes[airline_id].loc[
                    (airline_routes[airline_id]["origin"] == new_destination)
                    & (airline_routes[airline_id]["destination"] == new_origin)
                    & (airline_routes[airline_id]["fuel_stop"] == -1)
                ]
                updated_city_pair_out = city_pair_data.loc[
                    (city_pair_data["OriginCityID"] == new_origin)
                    & (city_pair_data["DestinationCityID"] == new_destination)
                ].iloc[0]
                updated_city_pair_in = city_pair_data.loc[
                    (city_pair_data["OriginCityID"] == new_destination)
                    & (city_pair_data["DestinationCityID"] == new_origin)
                ].iloc[0]
                # calculate sum of all seats that the airline has assigned to this itinerary including reassigned aircraft
                itin_seats = 0
                for ac_id in updated_new_itin_out["aircraft_ids"].iloc[0]:
                    itin_seats += aircraft_data.loc[airline_fleets[airline_id].loc[ac_id, "SizeClass"], "Seats"]
                rtn_flt_df.loc[
                    (rtn_flt_df["Origin"] == new_origin)
                    & (rtn_flt_df["Destination"] == new_destination)
                    & (rtn_flt_df["Fuel_Stop"] == -1),
                    "Profit_perSeat"
                ] = (
                    itin_profit(
                        updated_new_itin_out["fare"].iloc[0],
                        updated_new_itin_out.iloc[0],
                        updated_city_pair_out,
                        city_data.loc[new_origin],
                        city_data.loc[new_destination],
                        airline_fleets[airline_id],
                        aircraft_data,
                        # city_pair_data is already updated
                    ) + itin_profit(
                        updated_new_itin_in["fare"].iloc[0],
                        updated_new_itin_in.iloc[0],
                        updated_city_pair_in,
                        city_data.loc[new_destination],
                        city_data.loc[new_origin],
                        airline_fleets[airline_id],
                        aircraft_data,
                        # city_pair_data is already updated
                    )
                ) / itin_seats

                # move reassigned aircraft in rtn_flt_df
                rtn_flt_df.loc[
                    (rtn_flt_df["Origin"] == reassign_itin_out["origin"])
                    & (rtn_flt_df["Destination"] == reassign_itin_out["destination"])
                    & (rtn_flt_df["Fuel_Stop"] == reassign_itin_out["fuel_stop"]),
                    "Aircraft_IDs"
                ].iloc[0].remove(reassign_ac["AircraftID"])
                rtn_flt_df.loc[
                    (rtn_flt_df["Origin"] == new_origin)
                    & (rtn_flt_df["Destination"] == new_destination)
                    & (rtn_flt_df["Fuel_Stop"] == -1),
                    "Aircraft_IDs"
                ].iloc[0].append(int(reassign_ac["AircraftID"]))

                # if no planes left, remove itinerary from airline_routes and rtn_flt_df
                if len(airline_routes[airline_id].loc[out_reassign_mask, "aircraft_ids"].iloc[0]) == 0:
                    # remove itinerary from airline_routes
                    airline_routes[airline_id] = airline_routes[airline_id][~out_reassign_mask]
                    airline_routes[airline_id] = airline_routes[airline_id][~in_reassign_mask]
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

    return airline_routes, airline_fleets, city_pair_data, city_data
