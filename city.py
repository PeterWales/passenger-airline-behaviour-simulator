import pandas as pd
from operator import add
import numpy as np
import airline as al
import demand as demand
import aircraft as acft


def add_airports_to_cities(city_data: pd.DataFrame, airport_data: pd.DataFrame) -> tuple[pd.DataFrame, list[list]]:
    """
    Incorporate parts of airport_data into city_data DataFrame so that cities can be treated as single entities.
    Generate a list of lists for looking up which cities are located in which countries.

    city_data DataFrame is edited in-place to add the following columns:
        Population : int
            Population of the city in the current year
        Income_USDpercap : float
            Income per capita in the current year
        Latitude : float
            Capacity-weighted mean latitude of all airports in the city
        Longitude : float
            Capacity-weighted mean longitude of all airports in the city
        Domestic_Fees_USDperPax : float
            Capacity-weighted mean of domestic landing fees per passenger
        Domestic_Fees_USDperMovt : list[float]
            Capacity-weighted mean of domestic landing fees per landing for each aircraft size
        International_Fees_USDperPax : float
            Capacity-weighted mean of international landing fees per passenger
        International_Fees_USDperMovt : list[float]
            Capacity-weighted mean of international landing fees per landing for each aircraft size
        Taxi_Out_mins : float
            Capacity-weighted mean of unimpeded taxi-out time in minutes
        Taxi_In_mins : float
            Capacity-weighted mean of unimpeded taxi-in time in minutes
        Capacity_MovtsPerHr : float
            Total capacity of all airports in the city in movements per hour (takeoffs + landings)
        LongestRunway_m : float
            Length of the longest runway in the city

    city_lookup list is indexed to match CountryID field in CountryData file, and contains lists of CityIDs
    If a certain country doesn't contain any cities, the corresponding list element is empty

    Parameters
    ----------
    city_data : pd.DataFrame
    airport_data : pd.DataFrame

    Returns
    -------
    city_data : pd.DataFrame
    city_lookup : list[list]
    """
    # TODO: use a package for geodesic calculations

    # check number of different aircraft supported by airport_data
    n_aircraft = 0
    while (
        f"LandingCosts_PerMovt_Size{n_aircraft}_Domestic_2015USdollars"
        in airport_data.columns
    ):
        n_aircraft += 1
    if n_aircraft == 0:
        raise ValueError("No aircraft-specific data found in airport_data")

    # set CityID as index for faster lookups
    city_data.set_index('CityID', inplace=True)

    # set AirportID as index for faster lookups
    airport_data.set_index("AirportID", inplace=True)

    # initialise lists
    last_country_id = max(city_data["Country"])
    city_lookup = [[] for _ in range(last_country_id + 1)]
    latitude = []
    longitude = []
    domestic_fees_USDperpax = []
    domestic_fees_USDpermvmt = []
    international_fees_USDperpax = []
    international_fees_USDpermvmt = []
    taxi_out_mins = []
    taxi_in_mins = []
    capacity_perhr = []
    longest_runway_m = []

    # loop over rows of city_data
    for idx, city in city_data.iterrows():
        # calculate capacity-weighted mean of airport data
        capacity_sum = 0.0
        longest_runway_found = 0.0
        lat_sum = 0.0
        long_sum = 0.0
        dom_fee_pax_sum = 0.0
        dom_fee_mov_sum = [0.0] * n_aircraft
        intnl_fee_pax_sum = 0.0
        intnl_fee_mov_sum = [0.0] * n_aircraft
        taxi_out_sum = 0.0
        taxi_in_sum = 0.0
        airport_column = 1
        long_flag = False
        while f"Airport_{airport_column}" in city.index:
            airport_id = int(city[f"Airport_{airport_column}"])

            if not (airport_id == 0):
                airport = airport_data.loc[airport_id]

                capacity_sum += float(
                    airport["Capacity_movts_hr"]
                )

                longest_runway_found = max(
                    longest_runway_found,
                    float(airport["LongestRunway_m"]),
                )

                lat_sum += float(airport["Latitude"]) * float(
                    airport["Capacity_movts_hr"]
                )

                # check for edge case where a city has airports either side of the 180deg longitude line
                apt_longitude = float(airport["Longitude"])
                if (
                    apt_longitude * long_sum < 0.0
                    and (apt_longitude > 90.0 or apt_longitude < -90.0)
                ):
                    if apt_longitude < 0.0:
                        apt_longitude += 360.0
                    else:
                        apt_longitude -= 360.0
                    long_flag = True
                long_sum += apt_longitude * float(
                    airport["Capacity_movts_hr"]
                )

                dom_fee_pax_sum += float(
                    airport["LandingCosts_PerPax_Domestic_2015USdollars"]
                )
                dom_fee_mov_sum = list(
                    map(
                        add,
                        dom_fee_mov_sum,
                        [
                            float(
                                airport[
                                    f"LandingCosts_PerMovt_Size{ac}_Domestic_2015USdollars"
                                ]
                            )
                            for ac in range(n_aircraft)
                        ],
                    )
                )

                intnl_fee_pax_sum += float(
                    airport[
                        "LandingCosts_PerPax_International_2015USdollars"
                    ]
                )
                intnl_fee_mov_sum = list(
                    map(
                        add,
                        intnl_fee_mov_sum,
                        [
                            float(
                                airport[
                                    f"LandingCosts_PerMovt_Size{ac}_International_2015USdollars"
                                ]
                            )
                            for ac in range(n_aircraft)
                        ],
                    )
                )

                taxi_out_sum += float(
                    airport["UnimpTaxiOut_min"]
                )
                taxi_in_sum += float(
                    airport["UnimpTaxiIn_min"]
                )
            else:
                # airport_id == 0 => there are no more airports for that city
                break
            airport_column += 1

        if capacity_sum == 0.0:
            raise ValueError("City has no airport capacity")

        city_longitude = long_sum / capacity_sum
        if long_flag:
            if city_longitude < -180.0:
                city_longitude += 360.0
            if city_longitude > 180.0:
                city_longitude -= 360.0

        city_lookup[city["Country"]].append(idx)

        latitude.append(lat_sum / capacity_sum)
        longitude.append(city_longitude)
        domestic_fees_USDperpax.append(dom_fee_pax_sum / capacity_sum)
        domestic_fees_USDpermvmt.append(
            list(map(lambda x: x / capacity_sum, dom_fee_mov_sum))
        )
        international_fees_USDperpax.append(intnl_fee_pax_sum / capacity_sum)
        international_fees_USDpermvmt.append(
            list(map(lambda x: x / capacity_sum, intnl_fee_mov_sum))
        )
        taxi_out_mins.append(taxi_out_sum / capacity_sum)
        taxi_in_mins.append(taxi_in_sum / capacity_sum)
        capacity_perhr.append(capacity_sum)
        longest_runway_m.append(longest_runway_found)

    # add new columns to city_data
    city_data["Population"] = city_data["BaseYearPopulation"]
    city_data["Income_USDpercap"] = city_data["BaseYearIncome"]
    city_data["Latitude"] = latitude
    city_data["Longitude"] = longitude
    city_data["Domestic_Fees_USDperPax"] = domestic_fees_USDperpax
    city_data["Domestic_Fees_USDperMovt"] = domestic_fees_USDpermvmt
    city_data["International_Fees_USDperPax"] = international_fees_USDperpax
    city_data["International_Fees_USDperMovt"] = international_fees_USDpermvmt
    city_data["Taxi_Out_mins"] = taxi_out_mins
    city_data["Taxi_In_mins"] = taxi_in_mins
    city_data["Capacity_MovtsPerHr"] = capacity_perhr
    city_data["Movts_perHr"] = np.zeros(len(city_data))
    city_data["LongestRunway_m"] = longest_runway_m

    return city_data, city_lookup


def enforce_capacity(
    airlines: pd.DataFrame,
    airline_fleets: list[pd.DataFrame],
    airline_routes: list[pd.DataFrame],
    aircraft_data: pd.DataFrame,
    city_pair_data: pd.DataFrame,
    city_data: pd.DataFrame,
    city_lookup: list[list[int]],
    capacity_flag_list: list[int],
    demand_coefficients: dict[str, float]
) -> tuple[
    list[pd.DataFrame],
    list[pd.DataFrame],
    pd.DataFrame,
    pd.DataFrame
]:
    """
    Reassign aircraft to ensure city capacity limits are not exceeded

    Parameters
    ----------
    airlines : pd.DataFrame
        dataframe containing airline data
    airline_fleets : list[pd.DataFrame]
        list of dataframes containing airline fleet data
    airline_routes : list[pd.DataFrame]
        list of dataframes containing airline route data
    aircraft_data : pd.DataFrame
        dataframe containing aircraft data
    city_pair_data : pd.DataFrame
        dataframe containing route data
    city_data : pd.DataFrame
        dataframe containing city data
    city_lookup : list[list[int]]
        list of lists of city IDs by country code
    capacity_flag_list : list
        list of city IDs where capacity limits are exceeded
    demand_coefficients : dict
        dictionary of demand coefficients

    Returns
    -------
    airline_fleets : list[pd.DataFrame]
    airline_routes : list[pd.DataFrame]
    city_pair_data : pd.DataFrame
    city_data : pd.DataFrame
    """
    # TODO: - consider moving fuel stops away from capacity-constrained cities
    #       - add possibility of moving aircraft to routes that require fuel stops

    op_hrs_per_year = 6205  # =17*365 (assume airports are closed between 11pm and 6am)

    # iterate through all cities where total airport capacity is exceeded
    for city_id in capacity_flag_list:
        # initialise lists
        reassign_airline_id = []
        reassign_aircraft_ids = []
        reassign_destination = []
        reassign_profit = []
        reassign_out_fare = []
        reassign_in_fare = []
        reassign_out_time = []
        reassign_in_time = []
        reassign_flights_per_year = []
        reassign_fuel_stop = []

        for _, airline in airlines.iterrows():
            # find all routes that depart from city and are operated by airline (return flight then determined from outbound)
            city_airline_routes = airline_routes[airline["Airline_ID"]][
                airline_routes[airline["Airline_ID"]]["origin"] == city_id
            ]
            if city_airline_routes.empty:
                # airline doesn't operate to/from this city
                continue

            for __, out_itin in city_airline_routes.iterrows():
                city_pair = city_pair_data[
                    (city_pair_data["OriginCityID"] == out_itin["origin"])
                    & (city_pair_data["DestinationCityID"] == out_itin["destination"])
                ].iloc[0]
                origin = city_data.loc[out_itin["origin"]]
                destination = city_data.loc[out_itin["destination"]]

                in_itin = city_airline_routes[
                    (city_airline_routes["origin"] == out_itin["destination"])
                    & (city_airline_routes["destination"] == out_itin["origin"])
                ].iloc[0]

                # calculate profit of route (outbound + inbound)
                profit = al.itin_profit(
                    out_itin["fare"],
                    out_itin,
                    city_pair,
                    origin,
                    destination,
                    airline_fleets[airline["Airline_ID"]],
                    aircraft_data,
                    update_od_demand = False
                ) + al.itin_profit(
                    in_itin["fare"],
                    in_itin,
                    city_pair,
                    destination,
                    origin,
                    airline_fleets[airline["Airline_ID"]],
                    aircraft_data,
                    update_od_demand = False
                )

                # calculate flight time from aircraft speed and route distance
                flight_time_hrs = (
                    city_pair["Great_Circle_Distance_m"] / (
                        aircraft_data.at[airline_fleets[airline["Airline_ID"]].at[out_itin["aircraft_ids"][0], "SizeClass"], "CruiseV_ms"] * 3600
                    )
                )
                out_itin_time = flight_time_hrs + origin["Taxi_Out_mins"] + destination["Taxi_In_mins"]
                in_itin_time = flight_time_hrs + destination["Taxi_Out_mins"] + origin["Taxi_In_mins"]

                # save data
                reassign_airline_id.append(airline["Airline_ID"])
                reassign_aircraft_ids.append(out_itin["aircraft_ids"])
                reassign_destination.append(out_itin["destination"])
                reassign_profit.append(profit)
                reassign_out_fare.append(out_itin["fare"])
                reassign_in_fare.append(in_itin["fare"])
                reassign_out_time.append(out_itin_time)
                reassign_in_time.append(in_itin_time)
                reassign_flights_per_year.append(out_itin["flights_per_year"])
                reassign_fuel_stop.append(out_itin["fuel_stop"])

        # create dataframe and sort by profit (lowest first)
        potential_reassign = pd.DataFrame(
            {
                "Airline_ID": reassign_airline_id,
                "Aircraft_IDs": reassign_aircraft_ids,
                "Destination": reassign_destination,
                "Profit": reassign_profit,
                "Out_Fare": reassign_out_fare,
                "In_Fare": reassign_in_fare,
                "Out_Time": reassign_out_time,
                "In_Time": reassign_in_time,
                "Flights_perYear": reassign_flights_per_year,
                "Fuel_Stop": reassign_fuel_stop
            }
        )
        potential_reassign.sort_values("Profit", ascending=True, inplace=True)

        # reassign aircraft one-by-one until capacity is no longer exceeded, starting with lowest profit itinerary
        while city_data.loc[city_id, "Movts_perHr"] > city_data.loc[city_id, "Capacity_MovtsPerHr"]:
            # find which aircraft are assigned to this itinerary and extract their info from fleet_df
            aircraft_ids = potential_reassign.iloc[0]["Aircraft_IDs"]
            fleet_df = airline_fleets[potential_reassign.iloc[0]["Airline_ID"]]
            itin_ac = fleet_df[fleet_df["AircraftID"].isin(aircraft_ids)]

            # extract the largest aircraft for reassignment (tie break: take oldest aircraft first)
            itin_ac.sort_values("SizeClass", "Age_years", ascending=[False, False], inplace=True)
            reassign_ac = itin_ac.iloc[0]
            airline_id = itin_ac.iloc[0]["Airline_ID"]
            # drop reassign_ac from potential_reassign.iloc[0]["Aircraft_IDs"]
            potential_reassign.iloc[0]["Aircraft_IDs"].remove(reassign_ac["AircraftID"])

            # update movements in Origin, Destination and FuelStop city_data
            movts = 2*reassign_ac["Flights_perYear"] / op_hrs_per_year  # Flights_perYear is no. return flights
            city_data.loc[city_id, "Movts_perHr"] -= movts
            city_data.loc[reassign_ac["Destination"], "Movts_perHr"] -= movts
            if reassign_ac["FuelStop"] is not None:
                city_data.loc[reassign_ac["FuelStop"], "Movts_perHr"] -= movts

            # update seat_flights_per_year in city_pair_data
            outbound_mask = (
                (city_pair_data["OriginCityID"] == reassign_ac["RouteOrigin"])
                & (city_pair_data["DestinationCityID"] == reassign_ac["RouteDestination"])
            )
            inbound_mask = (
                (city_pair_data["OriginCityID"] == reassign_ac["RouteDestination"])
                & (city_pair_data["DestinationCityID"] == reassign_ac["RouteOrigin"])
            )
            seat_flights_per_year = reassign_ac["Flights_perYear"] * aircraft_data.at[reassign_ac["SizeClass"], "Seats"]
            city_pair_data.loc[outbound_mask, "seat_flights_per_year"] -= seat_flights_per_year
            city_pair_data.loc[inbound_mask, "seat_flights_per_year"] -= seat_flights_per_year

            # update utility in city_pair_data (delta utility is itin exp(utility) with new Flights_perYear - existing exp(utility) value)
            outbound_delta_exp_utility = (
                demand.calc_exp_utility(
                    demand_coefficients,
                    potential_reassign.iloc[0]["Out_Fare"],
                    potential_reassign.iloc[0]["Out_Time"],
                    potential_reassign.iloc[0]["Flights_perYear"] - reassign_ac["Flights_perYear"],
                    potential_reassign.iloc[0]["Fuel_Stop"],
                )  # will return 0 if Flights_perYear is 0
                - demand.calc_exp_utility(
                    demand_coefficients,
                    potential_reassign.iloc[0]["Out_Fare"],
                    potential_reassign.iloc[0]["Out_Time"],
                    potential_reassign.iloc[0]["Flights_perYear"],
                    potential_reassign.iloc[0]["Fuel_Stop"],
                )
            )
            inbound_delta_exp_utility = (
                demand.calc_exp_utility(
                    demand_coefficients,
                    potential_reassign.iloc[0]["In_Fare"],
                    potential_reassign.iloc[0]["In_Time"],
                    potential_reassign.iloc[0]["Flights_perYear"] - reassign_ac["Flights_perYear"],
                    potential_reassign.iloc[0]["Fuel_Stop"],
                )  # will return 0 if Flights_perYear is 0
                - demand.calc_exp_utility(
                    demand_coefficients,
                    potential_reassign.iloc[0]["In_Fare"],
                    potential_reassign.iloc[0]["In_Time"],
                    potential_reassign.iloc[0]["Flights_perYear"],
                    potential_reassign.iloc[0]["Fuel_Stop"],
                )
            )
            city_pair_data.loc[outbound_mask, "exp_utility_sum"] += outbound_delta_exp_utility
            city_pair_data.loc[inbound_mask, "exp_utility_sum"] += inbound_delta_exp_utility

            # reassign aircraft to route with highest base demand that isn't limited by capacity or aircraft range
            city_pair_data.sort_values("BaseYearODDemandPax_Est", ascending=False, inplace=True)
            assigned = False
            for _, city_pair in city_pair_data.iterrows():
                # check whether origin or destination are in country where airline is located
                if (
                    city_pair["OriginCityID"] in city_lookup[airline["CountryID"]]
                    or city_pair["DestinationCityID"] in city_lookup[airline["CountryID"]]
                ):
                    # check aircraft has enough range and runways are long enough
                    if (
                        city_pair["Great_Circle_Distance_m"] < aircraft_data.at[reassign_ac["SizeClass"], "TypicalRange_m"]
                        and city_data.loc[city_pair["OriginCityID"], "LongestRunway_m"] > aircraft_data.at[reassign_ac["SizeClass"], "TakeoffDist_m"]
                        and city_data.loc[city_pair["OriginCityID"], "LongestRunway_m"] > aircraft_data.at[reassign_ac["SizeClass"], "LandingDist_m"]
                        and city_data.loc[city_pair["DestinationCityID"], "LongestRunway_m"] > aircraft_data.at[reassign_ac["SizeClass"], "TakeoffDist_m"]
                        and city_data.loc[city_pair["DestinationCityID"], "LongestRunway_m"] > aircraft_data.at[reassign_ac["SizeClass"], "LandingDist_m"]
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
                            None
                        )
                        if (
                            city_data.loc[origin_id, "Movts_perHr"] + (2*flights_per_year/op_hrs_per_year) <= city_data.loc[origin_id, "Capacity_MovtsPerHr"]
                            and city_data.loc[destination_id, "Movts_perHr"] + (2*flights_per_year/op_hrs_per_year) <= city_data.loc[destination_id, "Capacity_MovtsPerHr"]
                        ):
                            return_mask = (
                                (city_pair_data["OriginCityID"] == destination_id)
                                & (city_pair_data["DestinationCityID"] == origin_id)
                                & (city_pair_data["FuelStopCityID"] == None)
                            )  # for city_pair_data (outbound flight can just use city_pair.name)

                            # update movements in city_data
                            city_data.loc[origin_id, "Movts_perHr"] += 2*flights_per_year/op_hrs_per_year  # 2 movmts per return flight (takeoff and landing)
                            city_data.loc[destination_id, "Movts_perHr"] += 2*flights_per_year/op_hrs_per_year
                            
                            # update seat_flights_per_year in city_pair_data
                            seat_flights_per_year = flights_per_year * aircraft_data.at[reassign_ac["SizeClass"], "Seats"]
                            city_pair_data.loc[city_pair.name, "seat_flights_per_year"] += seat_flights_per_year
                            city_pair_data.loc[return_mask, "seat_flights_per_year"] += seat_flights_per_year

                            # update exp(utility) in city_pair_data
                            # NOTE ["Fare_Est"] can still be used here since initialisation step - airlines haven't chosen their own fares yet
                            flight_time_hrs = city_pair["Great_Circle_Distance_m"] / (aircraft_data.at[reassign_ac["SizeClass"], "CruiseV_ms"] * 3600)
                            out_time = flight_time_hrs + city_data.loc[origin_id, "Taxi_Out_mins"] + city_data.loc[destination_id, "Taxi_In_mins"]
                            in_time = flight_time_hrs + city_data.loc[destination_id, "Taxi_Out_mins"] + city_data.loc[origin_id, "Taxi_In_mins"]
                            outbound_utility = demand.calc_exp_utility(
                                demand_coefficients,
                                city_pair["Fare_Est"],
                                out_time,
                                city_pair["seat_flights_per_year"],  # already added new flights
                                None
                            )
                            outbound_delta_exp_utility = (
                                outbound_utility - demand.calc_exp_utility(
                                    demand_coefficients,
                                    city_pair["Fare_Est"],
                                    out_time,
                                    city_pair["seat_flights_per_year"] - seat_flights_per_year,  # previous utility without new flights
                                    None
                                )
                            )
                            return_fare = city_pair_data.loc[return_mask, "Fare_Est"].iloc[0]
                            inbound_utility = demand.calc_exp_utility(
                                demand_coefficients,
                                return_fare,
                                in_time,
                                city_pair["seat_flights_per_year"],  # already added new flights
                                None
                            )
                            inbound_delta_exp_utility = (
                                inbound_utility - demand.calc_exp_utility(
                                    demand_coefficients,
                                    return_fare,
                                    in_time,
                                    city_pair["seat_flights_per_year"] - seat_flights_per_year,
                                    None
                                )
                            )
                            city_pair_data.loc[city_pair.name, "exp_utility_sum"] += outbound_delta_exp_utility
                            city_pair_data.loc[return_mask, "exp_utility_sum"] += inbound_delta_exp_utility

                            # add exp(utility), flights_per_year and seat_flights_per_year to airline_routes
                            if airline_routes[airline_id][
                                (airline_routes[airline_id]["origin"] == origin_id)
                                & (airline_routes[airline_id]["destination"] == destination_id)
                                & (airline_routes[airline_id]["fuel_stop"] == None)
                            ].empty:
                                # create new route
                                new_out_route = {
                                    "origin": [origin_id],
                                    "destination": [destination_id],
                                    "fare": [city_pair["Fare_Est"]],
                                    "aircraft_ids": [[reassign_ac["AircraftID"]]],
                                    "flights_per_year": [flights_per_year],
                                    "seat_flights_per_year": [seat_flights_per_year],
                                    "exp_utility": [outbound_utility]
                                }
                                new_in_route = {
                                    "origin": [destination_id],
                                    "destination": [origin_id],
                                    "fare": [return_fare],
                                    "aircraft_ids": [[reassign_ac["AircraftID"]]],
                                    "flights_per_year": [flights_per_year],
                                    "seat_flights_per_year": [seat_flights_per_year],
                                    "exp_utility": [inbound_utility]
                                }
                                new_out_df = pd.DataFrame(new_out_route)
                                new_in_df = pd.DataFrame(new_in_route)
                                airline_routes[airline_id] = pd.concat([
                                    airline_routes[airline_id],
                                    new_out_df,
                                    new_in_df
                                ], ignore_index=True)
                            else:
                                # add to existing route
                                out_mask = (
                                    (airline_routes[airline_id]["origin"] == origin_id)
                                    & (airline_routes[airline_id]["destination"] == destination_id)
                                    & (airline_routes[airline_id]["fuel_stop"] == None)
                                )
                                in_mask = (
                                    (airline_routes[airline_id]["origin"] == destination_id)
                                    & (airline_routes[airline_id]["destination"] == origin_id)
                                    & (airline_routes[airline_id]["fuel_stop"] == None)
                                )
                                airline_routes[airline_id].loc[out_mask, "aircraft_ids"].iloc[0].append(reassign_ac["AircraftID"])
                                airline_routes[airline_id].loc[in_mask, "aircraft_ids"].iloc[0].append(reassign_ac["AircraftID"])
                                airline_routes[airline_id].loc[out_mask, "flights_per_year"] += flights_per_year
                                airline_routes[airline_id].loc[in_mask, "flights_per_year"] += flights_per_year
                                airline_routes[airline_id].loc[out_mask, "seat_flights_per_year"] += seat_flights_per_year
                                airline_routes[airline_id].loc[in_mask, "seat_flights_per_year"] += seat_flights_per_year
                                airline_routes[airline_id].loc[out_mask, "exp_utility"] = outbound_utility
                                airline_routes[airline_id].loc[in_mask, "exp_utility"] = inbound_utility

                            # edit aircraft in airline_fleets
                            acft_mask = airline_fleets[airline_id]["AircraftID"] == reassign_ac["AircraftID"]
                            airline_fleets[airline_id].loc[acft_mask, "RouteOrigin"] = origin_id
                            airline_fleets[airline_id].loc[acft_mask, "RouteDestination"] = destination_id
                            airline_fleets[airline_id].loc[acft_mask, "FuelStop"] = None
                            airline_fleets[airline_id].loc[acft_mask, "Flights_perYear"] = flights_per_year

                            assigned = True
            if not assigned:
                # no suitable route found, ground aircraft
                acft_mask = airline_fleets[airline_id]["AircraftID"] == reassign_ac["AircraftID"]
                airline_fleets[airline_id].loc[acft_mask, "RouteOrigin"] = None
                airline_fleets[airline_id].loc[acft_mask, "RouteDestination"] = None
                airline_fleets[airline_id].loc[acft_mask, "FuelStop"] = None
                airline_fleets[airline_id].loc[acft_mask, "Flights_perYear"] = 0

            if len(potential_reassign.iloc[0]["Aircraft_IDs"]) == 0:
                # remove this itinerary from potential_reassign and airline_routes if no more aircraft
                out_route_mask = (
                    (airline_routes[airline_id]["origin"] == city_id)
                    & (airline_routes[airline_id]["destination"] == potential_reassign.iloc[0]["Destination"])
                )
                in_route_mask = (
                    (airline_routes[airline_id]["origin"] == potential_reassign.iloc[0]["Destination"])
                    & (airline_routes[airline_id]["destination"] == city_id)
                )
                airline_routes[airline_id] = airline_routes[airline_id].drop(
                    airline_routes[airline_id][out_route_mask].index
                )
                airline_routes[airline_id] = airline_routes[airline_id].drop(
                    airline_routes[airline_id][in_route_mask].index
                )
                potential_reassign.drop(potential_reassign.index[0], inplace=True)

    return airline_fleets, airline_routes, city_pair_data, city_data
