import pandas as pd
import numpy as np
import route
from tqdm import tqdm
import aircraft as acft
import demand
import math
from scipy.optimize import minimize_scalar


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
) -> pd.DataFrame:
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
    
    Returns
    -------
    airline_fleets : list[pd.DataFrame]
    airline_routes : list[pd.DataFrame]
    city_pair_data : pd.DataFrame
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
            "fare": pd.Series(dtype="float"),
            "aircraft_ids": pd.Series(dtype="object"),
            "flights_per_year": pd.Series(dtype="int"),
            "seat_flights_per_year": pd.Series(dtype="int"),
            "exp_utility": pd.Series(dtype="float"),
        }
    ) for _ in range(n_airlines)]

    # iterate over all airlines
    # show a progress bar because this step can take a while
    for _, airline in tqdm(
        airlines.iterrows(),
        total=airlines.shape[0],
        desc="        Airline fleets initialised",
        ascii=False,
        ncols=75,
    ):
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
            total_seats += n_aircraft_type * aircraft_data.at[aircraft_type, "Seats"]
            if (n_aircraft_type > 0) and (aircraft_data.at[aircraft_type, "Seats"] < smallest_ac):
                smallest_ac = aircraft_data.at[aircraft_type, "Seats"]

        aircraft_id_list = []
        aircraft_size_list = []
        aircraft_age_list = []
        current_lease_list = []
        breguet_factor_list = []
        origin_id_list = []
        destination_id_list = []
        fuel_stop_list = []
        flights_per_year_list = []

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

            for _, aircraft in aircraft_data.iterrows():
                if seats <= (smallest_ac * min_load_factor):
                    break
                aircraft_size = aircraft["AircraftID"]
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
                                fuel_stop = None

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
                                    airline_routes,
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

                            if fuel_stop is not None:  # None if route not possible wih a stop for that aircraft
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
                                        airline_routes,
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

    return airline_fleets, airline_routes, city_pair_data


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
    fuel_stop: int | None,
    flights_per_year_list: list,
    city_data: pd.DataFrame,
    city_pair_data: pd.DataFrame,
    airline_routes: list[pd.DataFrame],
    airline_id: int,
    outbound_route: pd.Series,
    inbound_route: pd.Series,
    demand_coefficients: dict[str, float],
):
    aircraft_id_list.append(aircraft_id)
    aircraft_size_list.append(aircraft_size)
    aircraft_age_list.append(randomGen.randint(0, aircraft["RetirementAge_years"]-1))
    current_lease_list.append(aircraft["LeaseRateNew_USDPerMonth"] * (aircraft["LeaseRateAnnualMultiplier"] ** aircraft_age_list[-1]))
    breguet_factor_list.append((aircraft["Breguet_gradient"] * year) + aircraft["Breguet_intercept"])
    origin_id_list.append(origin_id)
    destination_id_list.append(destination_id)
    fuel_stop_list.append(fuel_stop)

    if fuel_stop is None:
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

    seat_flights_per_year = flights_per_year_list[-1] * aircraft["Seats"]
    flight_time_hrs = (
        outbound_route["Great_Circle_Distance_m"] / (aircraft["CruiseV_ms"] * 3600)
    )  # could make this more accurate by taking average of different aircraft rather than just the last one
    outbound_exp_utility = demand.calc_exp_utility(
        demand_coefficients,
        outbound_route["Fare_Est"],
        flight_time_hrs,
        airline_routes[airline_id].loc[outbound_mask, "flights_per_year"] + flights_per_year_list[-1],
        fuel_stop,
    )
    inbound_exp_utility = demand.calc_exp_utility(
        demand_coefficients,
        inbound_route["Fare_Est"],
        flight_time_hrs,
        airline_routes[airline_id].loc[inbound_mask, "flights_per_year"] + flights_per_year_list[-1],
        fuel_stop,
    )

    # update outbound and return route in-place in city_pair_data DataFrame
    city_pair_data.loc[
        (city_pair_data["OriginCityID"] == origin_id)
        & (city_pair_data["DestinationCityID"] == destination_id),
        "seat_flights_per_year"
    ] += seat_flights_per_year
    city_pair_data.loc[
        (city_pair_data["OriginCityID"] == destination_id)
        & (city_pair_data["DestinationCityID"] == origin_id),
        "seat_flights_per_year"
    ] += seat_flights_per_year
    city_pair_data.loc[
        (city_pair_data["OriginCityID"] == origin_id)
        & (city_pair_data["DestinationCityID"] == destination_id),
        "exp_utility_sum"
    ] += outbound_exp_utility
    city_pair_data.loc[
        (city_pair_data["OriginCityID"] == destination_id)
        & (city_pair_data["DestinationCityID"] == origin_id),
        "exp_utility_sum"
    ] += inbound_exp_utility

    # update airline-specific route dataframe
    not_route_exists = airline_routes[airline_id][
        (airline_routes[airline_id]["origin"] == origin_id)
        & (airline_routes[airline_id]["destination"] == destination_id)
        & (airline_routes[airline_id]["fuel_stop"] == fuel_stop)  # treat as a seperate itinerary if fuel stop is different
    ].empty
    if not_route_exists:
        airline_routes_newrow_1 = {
            "origin": [origin_id],
            "destination": [destination_id],
            "fare": [outbound_route["Fare_Est"]],
            "aircraft_ids": [[aircraft_id]],
            "flights_per_year": [flights_per_year_list[-1]],
            "seat_flights_per_year": [seat_flights_per_year],
            "exp_utility": [outbound_exp_utility]
        }
        new_df_1 = pd.DataFrame(airline_routes_newrow_1)
        airline_routes_newrow_2 = {
            "origin": [destination_id],
            "destination": [origin_id],
            "fare": [inbound_route["Fare_Est"]],
            "aircraft_ids": [[aircraft_id]],
            "flights_per_year": [flights_per_year_list[-1]],
            "seat_flights_per_year": [seat_flights_per_year],
            "exp_utility": [inbound_exp_utility]
        }
        new_df_2 = pd.DataFrame(airline_routes_newrow_2)
        airline_routes[airline_id] = pd.concat([
            airline_routes[airline_id], 
            new_df_1,
            new_df_2
        ], ignore_index=True)
    else:
        outbound_mask = (
            (airline_routes[airline_id]["origin"] == origin_id) & 
            (airline_routes[airline_id]["destination"] == destination_id)
        )
        inbound_mask = (
            (airline_routes[airline_id]["origin"] == destination_id) & 
            (airline_routes[airline_id]["destination"] == origin_id)
        )

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
        airline_routes,
    )


def optimise_fares(
    airlines: pd.DataFrame,
    airline_routes: list[pd.DataFrame],
    airline_fleets: list[pd.DataFrame],
    city_pair_data: pd.DataFrame,
    city_data: pd.DataFrame,
    aircraft_data: pd.DataFrame,
    maxiters: int,
    demand_tolerance: float,
) -> tuple[list[pd.DataFrame], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Optimise fares for all airlines, assuming no change in flight schedules
    """
    # initialise dataframes tracking convergence
    fare_iters = pd.DataFrame()
    fare_iters["base"] = city_pair_data["Mean_Fare_USD"]
    demand_iters = pd.DataFrame()
    demand_iters["base"] = city_pair_data["Total_Demand"]

    for iteration in range(maxiters):
        # allow each airline to adjust their prices without knowledge of other airlines' choices
        for airline in airlines:
            for itin in airline_routes[airline]:
                city_pair = city_pair_data[
                    (city_pair_data["OriginCityID"] == itin["origin"])
                    & (city_pair_data["DestinationCityID"] == itin["destination"])
                ].iloc[0]
                old_fare = itin["fare"]
                itin["fare"] = maximise_itin_profit(
                    itin,
                    airline_fleets[airline],
                    city_pair_data,
                    city_data,
                    aircraft_data,
                )

                # update city_pair_data with new mean fare
                itin_fare_diff = itin["fare"] - old_fare
                if itin_fare_diff != 0:
                    prev_mean_fare = city_pair["Mean_Fare_USD"]
                    city_pair["Mean_Fare_USD"] = (
                        (prev_mean_fare * city_pair["seat_flights_per_year"])
                        + (itin_fare_diff * itin["flights_per_year"])
                    ) / city_pair["seat_flights_per_year"]

        # update demand for all O-D pairs
        for city_pair in city_pair_data:
            city_pair["Total_Demand"] = demand.update_od_demand(city_pair)

        # check convergence
        fare_iters[f"iter{iteration}"] = city_pair_data["Mean_Fare_USD"]
        demand_iters[f"iter{iteration}"] = city_pair_data["Total_Demand"]
        if (abs(demand_iters[f"iter{iteration}"] - demand_iters[f"iter{iteration-1}"]) < demand_tolerance).all():
            break

    return airline_routes, city_pair_data, fare_iters, demand_iters


def maximise_itin_profit(
    airline_route: pd.Series,
    fleet_df: pd.DataFrame,
    city_pair_data: pd.DataFrame,
    city_data: pd.DataFrame,
    aircraft_data: pd.DataFrame,
) -> float:
    """
    Maximise profit for a given itinerary by adjusting fare
    """
    fare_bounds = (0, 50000)

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
            update_od_demand=False
        )

    result = minimize_scalar(objective, bounds=fare_bounds, method="bounded")
    optimal_fare = result.x
    return optimal_fare


def itin_profit(
    fare: float,
    airline_route: pd.Series,
    city_pair: pd.Series,
    origin: pd.Series,
    destination: pd.Series,
    fleet_data: pd.DataFrame,
    aircraft_data: pd.DataFrame,
    update_od_demand : bool = False
):
    """
    Calculate profit for a given itinerary (one-way) and fare
    """
    # TODO: implement variation in fuel price over time

    # NOTE since this function acts on rows of DataFrames, it does not alter the original DataFrames

    FuelCost_USDperGallon = 1.5  # PLACEHOLDER

    # update city_pair mean fare (weighted by seats available not seats filled)
    if update_od_demand:
        total_revenue = city_pair["Mean_Fare_USD"] * city_pair["seat_flights_per_year"]
        total_revenue += (fare - airline_route["fare"] * airline_route["flights_per_year"])
        city_pair["Mean_Fare_USD"] = total_revenue / city_pair["seat_flights_per_year"]
        city_pair["Total_Demand"] = demand.update_od_demand(city_pair)

    # update airline route fare
    airline_route["fare"] = fare

    # can't sell more tickets than the airline has scheduled
    annual_itin_demand = demand.update_itinerary_demand(city_pair, airline_route), airline_route["seat_flights_per_year"]
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
