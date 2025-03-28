import pandas as pd
import numpy as np
import route
import aircraft as acft
import demand
import reassignment
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
                x[1]["BaseYearGDP"],
                x[1]["BaseYearPopulation"],
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
    airlines["Grounded_acft"] = [[] for _ in range(len(airlines))]

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

    # add movements to cities and flag if capacity limit exceeded
    city_data.loc[origin_id, "Movts_perHr"] += 2.0 * float(flights_per_year_list[-1]) / op_hrs_per_year  # 2* since each flight is return
    city_data.loc[destination_id, "Movts_perHr"] += 2.0 * float(flights_per_year_list[-1]) / op_hrs_per_year
    
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
        city_data.loc[fuel_stop, "Movts_perHr"] += 4.0 * float(flights_per_year_list[-1]) / op_hrs_per_year  # 4* since 1 takeoff and 1 landing for each leg
        if (
            (city_data.loc[fuel_stop, "Movts_perHr"] > city_data.loc[fuel_stop, "Capacity_MovtsPerHr"])
            and (fuel_stop not in capacity_flag_list)
        ):
            capacity_flag_list.append(fuel_stop)

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
    FuelCost_USDperGallon,
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

    Returns
    -------
    airline_routes : list[pd.DataFrame]
    city_pair_data : pd.DataFrame
    """
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
                max_fare,
                FuelCost_USDperGallon,
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

    return airline_routes, city_pair_data


def maximise_itin_profit(
    airline_route: pd.Series,
    fleet_df: pd.DataFrame,
    city_pair_data: pd.DataFrame,
    city_data: pd.DataFrame,
    aircraft_data: pd.DataFrame,
    max_fare: float,
    FuelCost_USDperGallon: float,
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
            FuelCost_USDperGallon,
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
    FuelCost_USDperGallon: float,
    add_city_pair_seat_flights: int | None = None,
    add_city_pair_exp_utility: float | None = None,
    add_itin_seat_flights: int | None = None,
    add_itin_exp_utility: float | None = None
) -> float:
    """
    Calculate profit for a given itinerary (one-way) and fare
    """
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

    # update city_pair mean fare (weighted by seats available not seats filled)
    total_revenue = city_pair["Mean_Fare_USD"] * city_pair["Seat_Flights_perYear"]
    total_revenue += ((fare - airline_route["fare"]) * airline_route["seat_flights_per_year"])
    city_pair["Mean_Fare_USD"] = total_revenue / city_pair["Seat_Flights_perYear"]
    city_pair["Total_Demand"] = demand.update_od_demand(city_pair)

    # update airline route fare
    old_fare = airline_route["fare"]
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

    # reset dataframes to avoid mutability issues
    airline_route["fare"] = old_fare
    if add_city_pair_seat_flights is not None:
        city_pair["Seat_Flights_perYear"] -= add_city_pair_seat_flights
    if add_city_pair_exp_utility is not None:
        city_pair["Exp_Utility_Sum"] -= add_city_pair_exp_utility
    if add_itin_seat_flights is not None:
        airline_route["seat_flights_per_year"] -= add_itin_seat_flights
    if add_itin_exp_utility is not None:
        airline_route["exp_utility"] -= add_itin_exp_utility
    
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

    op_hrs_per_year = 6205.0  # airport op hours per year = 17*365 (assume airports are closed between 11pm and 6am)

    # iterate over all airlines
    for _, airline in airlines.iterrows():
        print(f"    Reassigning aircraft for airline {airline['Airline_ID']}")

        airline_id = airline["Airline_ID"]

        # create dataframe of airline's existing routes and their profit per seat
        rtn_flt_df = reassignment.calc_existing_profits(
            airline_routes[airline_id],
            city_data,
            city_pair_data,
            aircraft_data,
            airline_fleets[airline_id],
            FuelCost_USDperGallon,
        )

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
            itin_ac = airline_fleets[airline_id][airline_fleets[airline_id]["AircraftID"].isin(itin_aircraft_ids)]
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
            ) = reassignment.best_itin_alternative(
                city_data,
                city_pair_data,
                city_lookup,
                airlines.loc[airline_id, "CountryID"],
                airline_routes[airline_id],
                airline_fleets[airline_id],
                aircraft_data,
                reassign_ac,
                op_hrs_per_year,
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
                    airline_id,
                    airline_fleets,
                    aircraft_data,
                    demand_coefficients,
                    op_hrs_per_year,
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
                )

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
            
        # deal with grounding and leases where appropriate
        rtn_flt_df.sort_values("Profit_perSeat", ascending=True, inplace=True)
        # check whether any itineraries are making negative profit
        if rtn_flt_df["Profit_perSeat"].iloc[0] < 0.0:
            # end lease on all currently grounded aircraft
            for ac_idx in airlines.loc[airline_id, "Grounded_acft"]:
                # remove aircraft from airline_fleets
                airline_fleets[airline_id] = airline_fleets[airline_id][
                    airline_fleets[airline_id]["AircraftID"] != ac_idx
                ]
            airlines.loc[airline_id, "Grounded_acft"] = []

            # ground aircraft that are making negative profit
            while rtn_flt_df["Profit_perSeat"].iloc[0] < 0.0:
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

                # choose largest aircraft on route to reassign
                itin_aircraft_ids = reassign_row["Aircraft_IDs"]
                itin_ac = airline_fleets[airline_id][airline_fleets[airline_id]["AircraftID"].isin(itin_aircraft_ids)]
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
                (
                    airline_routes,
                    airline_fleets,
                    city_data,
                    city_pair_data,
                ) = reassignment.reassign_ac_to_new_route(
                    -1,
                    -1,
                    out_reassign_mask,
                    in_reassign_mask,
                    reassign_itin_out,
                    reassign_itin_in,
                    reassign_ac,
                    0,
                    0,
                    city_pair_data,
                    city_data,
                    airline_routes,
                    airline_id,
                    airline_fleets,
                    aircraft_data,
                    demand_coefficients,
                    op_hrs_per_year,
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
                )

                # reorder rtn_flt_df
                rtn_flt_df.sort_values("Profit_perSeat", ascending=True, inplace=True)


                

    return airline_routes, airline_fleets, city_pair_data, city_data
