from dataclasses import dataclass
import pandas as pd
import numpy as np
import route


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
    aircraft = []
    airline_routes = []

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
                    aircraft.append([])
                    airline_routes.append(
                        pd.DataFrame(
                            columns=[
                                "origin",
                                "destination",
                                "fare",
                                "aircraft_ids",
                                "seat_flights_per_year",
                            ]
                        )
                    )
                    airline_idx += 1

    airlines["Airline_ID"] = airline_id
    airlines["Region"] = region_id
    airlines["Country"] = country_name
    airlines["CountryID"] = country_id
    airlines["n_Aircraft"] = aircraft_lists
    airlines["Aircraft"] = aircraft
    airlines["Airline_Routes"] = airline_routes

    return airlines


def initialise_fleet_assignment(
    airlines: pd.DataFrame,
    city_pair_data: pd.DataFrame,
    city_data: pd.DataFrame,
    aircraft_data: pd.DataFrame,
    city_lookup: list,
    randomGen: np.random.Generator,
    year: int,
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
    airlines : pd.DataFrame
    city_pair_data : pd.DataFrame
    """
    # TODO: enable EU airlines to operate routes between all EU countries
    # TODO: move aircraft creation and following lines into a function to avoid code duplication

    min_load_factor = 0.8  # minimum load factor for an aircraft to be assigned to a route

    n_cities = len(city_pair_data)
    n_airlines = len(airlines)

    # create a list of airline fleet DataFrames
    airline_fleets = [pd.DataFrame() for _ in range(n_airlines)]
    airline_routes = [pd.DataFrame(
        columns=[
            "origin",
            "destination",
            "fare",
            "aircraft_ids",
            "seat_flights_per_year",
        ]
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
        for aircraft_type, n_aircraft_type in enumerate(airline["n_Aircraft"]):
            total_seats += n_aircraft_type * aircraft_data.at[aircraft_type, "Seats"]

        aircraft_id_list = []
        aircraft_size_list = []
        aircraft_age = []
        current_lease = []
        breguet_factor = []
        origin_id_list = []
        destination_id_list = []
        fuel_stop_list = []
        flights_per_year = []

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
                if seats <= 0:
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

                                aircraft_id_list.append(aircraft_id)
                                aircraft_size_list.append(aircraft_size)
                                aircraft_age.append(randomGen.randint(0, aircraft["RetirementAge_years"]-1))
                                current_lease.append(aircraft["LeaseRateNew_USDPerMonth"] * (aircraft["LeaseRateAnnualMultiplier"] ** aircraft_age[-1]))
                                breguet_factor.append((aircraft["Breguet_gradient"] * year) + aircraft["Breguet_intercept"])
                                origin_id_list.append(origin_id)
                                destination_id_list.append(destination_id)
                                fuel_stop_list.append(None)

                                flights_per_year.append(
                                    aircraft.calc_flights_per_year(
                                        city_data.loc[origin_id].iloc[0],
                                        city_data.loc[destination_id].iloc[0],
                                        aircraft,
                                        city_pair_data,
                                        fuel_stop=None,
                                    )
                                )

                                # update outbound and return route in-place in city_pair_data DataFrame
                                city_pair_data.loc[
                                    (city_pair_data["OriginCityID"] == origin_id)
                                    & (city_pair_data["DestinationCityID"] == destination_id),
                                    "SeatFlightsPerYear"
                                ] += flights_per_year[-1] * aircraft["Seats"]
                                city_pair_data.loc[
                                    (city_pair_data["OriginCityID"] == destination_id)
                                    & (city_pair_data["DestinationCityID"] == origin_id),
                                    "SeatFlightsPerYear"
                                ] += flights_per_year[-1] * aircraft["Seats"]

                                # update airline-specific route dataframe
                                not_route_exists = airline_routes[airline_id][
                                    (airline_routes[airline_id]["origin"] == origin_id)
                                    & (airline_routes[airline_id]["destination"] == destination_id)
                                ].empty
                                if not_route_exists:
                                    airline_routes_row1 = {
                                        "origin": origin_id,
                                        "destination": destination_id,
                                        "fare": outbound_route["Fare_Est"],
                                        "aircraft_ids": [aircraft_id],
                                        "seat_flights_per_year": flights_per_year[-1] * aircraft["Seats"],
                                    },
                                    airline_routes_row2 = {
                                        "origin": destination_id,
                                        "destination": origin_id,
                                        "fare": inbound_route["Fare_Est"],
                                        "aircraft_ids": [aircraft_id],
                                        "seat_flights_per_year": flights_per_year[-1] * aircraft["Seats"],
                                    },
                                    airline_routes_len = len(airline_routes[airline_id])
                                    airline_routes[airline_id].loc[airline_routes_len] = airline_routes_row1
                                    airline_routes[airline_id].loc[airline_routes_len+1] = airline_routes_row2
                                else:
                                    airline_routes[airline_id].loc[
                                        (airline_routes[airline_id]["origin"] == origin_id)
                                        & (airline_routes[airline_id]["destination"] == destination_id),
                                        "aircraft_ids"
                                    ].append(aircraft_id)
                                    airline_routes[airline_id].loc[
                                        (airline_routes[airline_id]["origin"] == origin_id)
                                        & (airline_routes[airline_id]["destination"] == destination_id),
                                        "seat_flights_per_year"
                                    ] += flights_per_year[-1] * aircraft["Seats"]

                        elif distance < 2*aircraft["TypicalRange_m"]:
                            # aircraft doesn't have enough range - try with a fuel stop
                            fuel_stop = route.choose_fuel_stop(
                                city_data,
                                city_data.loc[origin_id].iloc[0],
                                city_data.loc[destination_id].iloc[0],
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

                                    aircraft_id_list.append(aircraft_id)
                                    aircraft_size_list.append(aircraft_size)
                                    aircraft_age.append(randomGen.randint(0, aircraft["RetirementAge_years"]-1))
                                    current_lease.append(aircraft["LeaseRateNew_USDPerMonth"] * (aircraft["LeaseRateAnnualMultiplier"] ** aircraft_age[-1]))
                                    breguet_factor.append((aircraft["Breguet_gradient"] * year) + aircraft["Breguet_intercept"])
                                    origin_id_list.append(origin_id)
                                    destination_id_list.append(destination_id)
                                    fuel_stop_list.append(fuel_stop)

                                    flights_per_year.append(
                                        aircraft.calc_flights_per_year(
                                            city_data.loc[origin_id].iloc[0],
                                            city_data.loc[destination_id].iloc[0],
                                            aircraft,
                                            city_pair_data,
                                            fuel_stop,
                                        )
                                    )

                                    # update outbound and return route in-place in city_pair_data DataFrame
                                    city_pair_data.loc[
                                        (city_pair_data["OriginCityID"] == origin_id)
                                        & (city_pair_data["DestinationCityID"] == destination_id),
                                        "SeatFlightsPerYear"
                                    ] += flights_per_year[-1] * aircraft["Seats"]
                                    city_pair_data.loc[
                                        (city_pair_data["OriginCityID"] == destination_id)
                                        & (city_pair_data["DestinationCityID"] == origin_id),
                                        "SeatFlightsPerYear"
                                    ] += flights_per_year[-1] * aircraft["Seats"]

                                    # update airline-specific route dataframe
                                    not_route_exists = airline_routes[airline_id][
                                        (airline_routes[airline_id]["origin"] == origin_id)
                                        & (airline_routes[airline_id]["destination"] == destination_id)
                                    ].empty
                                    if not_route_exists:
                                        airline_routes_row1 = {
                                            "origin": origin_id,
                                            "destination": destination_id,
                                            "fare": outbound_route["Fare_Est"],
                                            "aircraft_ids": [aircraft_id],
                                            "seat_flights_per_year": flights_per_year[-1] * aircraft["Seats"],
                                        },
                                        airline_routes_row2 = {
                                            "origin": destination_id,
                                            "destination": origin_id,
                                            "fare": inbound_route["Fare_Est"],
                                            "aircraft_ids": [aircraft_id],
                                            "seat_flights_per_year": flights_per_year[-1] * aircraft["Seats"],
                                        },
                                        airline_routes_len = len(airline_routes[airline_id])
                                        airline_routes[airline_id].loc[airline_routes_len] = airline_routes_row1
                                        airline_routes[airline_id].loc[airline_routes_len+1] = airline_routes_row2
                                    else:
                                        airline_routes[airline_id].loc[
                                            (airline_routes[airline_id]["origin"] == origin_id)
                                            & (airline_routes[airline_id]["destination"] == destination_id),
                                            "aircraft_ids"
                                        ].append(aircraft_id)
                                        airline_routes[airline_id].loc[
                                            (airline_routes[airline_id]["origin"] == origin_id)
                                            & (airline_routes[airline_id]["destination"] == destination_id),
                                            "seat_flights_per_year"
                                        ] += flights_per_year[-1] * aircraft["Seats"]

                        else:
                            break
        
        fleet_df = pd.DataFrame()

        fleet_df["AircraftID"] = aircraft_id_list
        fleet_df["SizeClass"] = aircraft_size_list
        fleet_df["Age_years"] = aircraft_age
        fleet_df["Lease_USDperMonth"] = current_lease
        fleet_df["BreguetFactor"] = breguet_factor
        fleet_df["RouteOrigin"] = origin_id_list
        fleet_df["RouteDestination"] = destination_id_list
        fleet_df["FuelStop"] = fuel_stop_list
        fleet_df["Flights_perYear"] = flights_per_year

        airline_fleets[airline_id] = fleet_df

    return airline_fleets, airline_routes, airlines, city_pair_data
