import pandas as pd
from operator import add
import math
import numpy as np
import airline as al
import demand as demand
import aircraft as acft
import reassignment


def set_base_values(
    country_data: pd.DataFrame,
    population_data: pd.DataFrame,
    income_data: pd.DataFrame,
    base_year: int,
) -> pd.DataFrame:
    country_populations = []
    country_incomes = []
    for _, country in country_data.iterrows():
        country_id = country["Number"]
        country_populations.append(population_data.loc[population_data["Number"] == country_id, str(base_year)].iloc[0])
        country_incomes.append(income_data.loc[income_data["Number"] == country_id, str(base_year)].iloc[0])
    country_data["BaseYearPopulation"] = country_populations
    country_data["BaseYearGDP"] = country_incomes
    return country_data


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
    demand_coefficients: dict[str, float],
    FuelCost_USDperGallon: float,
) -> tuple[
    list[pd.DataFrame],
    list[pd.DataFrame],
    pd.DataFrame,
    pd.DataFrame
]:
    """
    Reassign aircraft from itineraries that are exceeding city capacity to itineraries that are not.

    Parameters
    ----------
    airlines : pd.DataFrame
    airline_fleets : list[pd.DataFrame]
    airline_routes : list[pd.DataFrame]
    aircraft_data : pd.DataFrame
    city_pair_data : pd.DataFrame
    city_data : pd.DataFrame
    city_lookup : list[list[int]]
    capacity_flag_list : list[int]
    demand_coefficients : dict[str, float]
    FuelCost_USDperGallon : float
    
    Returns
    -------
    airline_fleets : list[pd.DataFrame]
    airline_routes : list[pd.DataFrame]
    city_pair_data : pd.DataFrame
    city_data : pd.DataFrame
    airlines : pd.DataFrame
    """
    # TODO: - add possibility of moving aircraft to routes that require fuel stops
    #       - use reassignment.best_itin_alternative() to improve choice of itin to move reassigned aircraft to
    #       - recalculate profits for all airlines after reassignment, not just for airline which moves aircraft

    op_hrs_per_year = 6205.0  # airport op hours per year = 17*365 (assume airports are closed between 11pm and 6am)

    # iterate through all cities where total airport capacity is exceeded
    for city_id in capacity_flag_list:
        # iterate through all airlines
        for airline_id, airline in airlines.iterrows():
            # find all routes that depart from, arrive at, or stop for fuel at city and are operated by airline (return flight then determined from outbound)
            city_airline_routes = airline_routes[airline_id][
                (airline_routes[airline_id]["origin"] == city_id)
                | (airline_routes[airline_id]["destination"] == city_id)
                | (airline_routes[airline_id]["fuel_stop"] == city_id)
            ]  # note outbound and return itineraries are combined in reassignment.calc_existing_profits(), not here

            if city_airline_routes.empty:
                # airline doesn't operate any routes involving this city
                continue

            # find profit per seat for all airline's routes involving this city
            airline_reassign_df = reassignment.calc_existing_profits(
                city_airline_routes,
                city_data,
                city_pair_data,
                aircraft_data,
                airline_fleets[airline_id],
                FuelCost_USDperGallon,
                demand_coefficients,
            )
            airline_reassign_df["Airline_ID"] = airline_id

            # collate profit data for itineraries operated by all airlines which involve the capacity-limited city
            if "potential_reassign" in locals():
                potential_reassign = pd.concat(
                    [potential_reassign, airline_reassign_df],
                    ignore_index=True,
                )
            else:
                potential_reassign = airline_reassign_df

        # reassign aircraft one-by-one until capacity is no longer exceeded, starting with lowest profit itinerary
        while city_data.loc[city_id, "Movts_perHr"] > city_data.loc[city_id, "Capacity_MovtsPerHr"]:
            # sort all itineraries by profit (lowest first)
            potential_reassign.sort_values("Profit_perSeat", ascending=True, inplace=True)
            reassign_itin = potential_reassign.iloc[0]

            # find which aircraft are assigned to the lowest profit itinerary
            airline_id = int(reassign_itin["Airline_ID"])
            aircraft_ids = reassign_itin["Aircraft_IDs"]
            fleet_df = airline_fleets[airline_id]
            itin_ac = fleet_df[fleet_df["AircraftID"].isin(aircraft_ids)].copy(deep=True)

            # extract the smallest aircraft for reassignment (tie break: take oldest aircraft first)
            itin_ac.sort_values(["SizeClass", "Age_years"], ascending=[True, False], inplace=True)
            reassign_ac = itin_ac.iloc[0]  # smallest aircraft only

            out_reassign_mask = (
                (airline_routes[airline_id]["origin"] == reassign_itin["Origin"])
                & (airline_routes[airline_id]["destination"] == reassign_itin["Destination"])
                & (airline_routes[airline_id]["fuel_stop"] == reassign_itin["Fuel_Stop"])
            )
            in_reassign_mask = (
                (airline_routes[airline_id]["origin"] == reassign_itin["Destination"])
                & (airline_routes[airline_id]["destination"] == reassign_itin["Origin"])
                & (airline_routes[airline_id]["fuel_stop"] == reassign_itin["Fuel_Stop"])
            )

            # reassign aircraft to route with highest base demand that isn't limited by capacity or aircraft range
            city_pair_data.sort_values("BaseYearODDemandPax_Est", ascending=False, inplace=True)
            assigned = False
            for _, city_pair in city_pair_data.iterrows():
                # check whether origin or destination are in country where airline is located
                airline_country = airline["CountryID"]
                new_origin = city_pair["OriginCityID"]
                new_destination = city_pair["DestinationCityID"]
                if (
                    new_origin in city_lookup[airline_country]
                    or new_destination in city_lookup[airline_country]
                ):
                    # check aircraft has enough range and runways are long enough
                    if (
                        city_pair["Great_Circle_Distance_m"] < aircraft_data.loc[reassign_ac["SizeClass"], "TypicalRange_m"]
                        and city_data.loc[new_origin, "LongestRunway_m"] > aircraft_data.loc[reassign_ac["SizeClass"], "TakeoffDist_m"]
                        and city_data.loc[new_origin, "LongestRunway_m"] > aircraft_data.loc[reassign_ac["SizeClass"], "LandingDist_m"]
                        and city_data.loc[new_destination, "LongestRunway_m"] > aircraft_data.loc[reassign_ac["SizeClass"], "TakeoffDist_m"]
                        and city_data.loc[new_destination, "LongestRunway_m"] > aircraft_data.loc[reassign_ac["SizeClass"], "LandingDist_m"]
                    ):
                        # check adding aircraft won't exceed either city's capacity
                        flights_per_year = acft.calc_flights_per_year(
                            city_data.loc[new_origin],
                            new_origin,
                            city_data.loc[new_destination],
                            new_destination,
                            aircraft_data.loc[reassign_ac["SizeClass"]],
                            city_pair_data,
                            None,
                            -1
                        )
                        if (
                            city_data.loc[new_origin, "Movts_perHr"] + (2*float(flights_per_year)/op_hrs_per_year) <= city_data.loc[new_origin, "Capacity_MovtsPerHr"]
                            and city_data.loc[new_destination, "Movts_perHr"] + (2*float(flights_per_year)/op_hrs_per_year) <= city_data.loc[new_destination, "Capacity_MovtsPerHr"]
                        ):
                            # new itinerary is possible
                            addnl_seat_flights_per_year = flights_per_year * aircraft_data.loc[reassign_ac["SizeClass"], "Seats"]
                            
                            assigned = True
                            break
            
            if not assigned:
                # ground aircraft
                new_origin = -1
                new_destination = -1
                flights_per_year = 0
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
                airline_routes[airline_id].loc[out_reassign_mask].iloc[0],
                airline_routes[airline_id].loc[in_reassign_mask].iloc[0],
                reassign_ac,
                flights_per_year,
                addnl_seat_flights_per_year,
                city_pair_data,
                city_data,
                airline_routes,
                airlines,
                airline_id,
                airline_fleets,
                aircraft_data,
                demand_coefficients,
                op_hrs_per_year,
            )

            # split airline's itineraries out of potential_reassign
            airline_reassign_df = potential_reassign[potential_reassign["Airline_ID"] == airline_id].copy()
            potential_reassign.drop(potential_reassign[potential_reassign["Airline_ID"] == airline_id].index, inplace=True)

            # update masks because an itinerary may have been added to airline_routes
            out_reassign_mask = (
                (airline_routes[airline_id]["origin"] == reassign_itin["Origin"])
                & (airline_routes[airline_id]["destination"] == reassign_itin["Destination"])
                & (airline_routes[airline_id]["fuel_stop"] == reassign_itin["Fuel_Stop"])
            )
            in_reassign_mask = (
                (airline_routes[airline_id]["origin"] == reassign_itin["Destination"])
                & (airline_routes[airline_id]["destination"] == reassign_itin["Origin"])
                & (airline_routes[airline_id]["fuel_stop"] == reassign_itin["Fuel_Stop"])
            )

            # update profit per seat
            airline_reassign_df = reassignment.update_profit_tracker(
                airline_reassign_df,
                new_origin,
                new_destination,
                out_reassign_mask,
                in_reassign_mask,
                airline_routes[airline_id].loc[out_reassign_mask].iloc[0],
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

            # # add airline ID to any new rows
            # airline_reassign_df["Airline_ID"] = airline_id

            # account for the fact that reassignment.update_profit_tracker() can add new itineraries to potential_reassign
            airline_reassign_df = airline_reassign_df[
                (airline_reassign_df["Origin"] == city_id)
                | (airline_reassign_df["Destination"] == city_id)
                | (airline_reassign_df["Fuel_Stop"] == city_id)
            ]

            # add airline's itineraries back into potential_reassign
            potential_reassign = pd.concat(
                [potential_reassign, airline_reassign_df],
                ignore_index=True,
            )

            # if no planes left, remove itinerary from airline_routes and potential_reassign
            if len(airline_routes[airline_id].loc[out_reassign_mask, "aircraft_ids"].iloc[0]) == 0:
                # remove itinerary from airline_routes
                airline_routes[airline_id] = airline_routes[airline_id].loc[~out_reassign_mask]
                in_reassign_mask = (
                    (airline_routes[airline_id]["origin"] == reassign_itin["Destination"])
                    & (airline_routes[airline_id]["destination"] == reassign_itin["Origin"])
                    & (airline_routes[airline_id]["fuel_stop"] == reassign_itin["Fuel_Stop"])
                )  # recalculate mask since outbound itinerary has been removed
                airline_routes[airline_id] = airline_routes[airline_id].loc[~in_reassign_mask]
                # remove itinerary from rtn_flt_df
                potential_reassign = potential_reassign.drop(
                    potential_reassign[
                        (potential_reassign["Origin"] == reassign_itin["Origin"]) &
                        (potential_reassign["Destination"] == reassign_itin["Destination"]) &
                        (potential_reassign["Fuel_Stop"] == reassign_itin["Fuel_Stop"])
                    ].index
                )

    return airline_fleets, airline_routes, city_pair_data, city_data, airlines


def annual_update(
    country_data: pd.DataFrame,
    city_data: pd.DataFrame,
    city_lookup: list,
    population_data: pd.DataFrame,
    income_data: pd.DataFrame,
    year_entering: int,
):
    for _, country in country_data.iterrows():
        country_pop = population_data[population_data["Number"] == country["Number"]]
        country_inc = income_data[income_data["Number"] == country["Number"]]

        pop_multiplier = country_pop[str(year_entering)].iloc[0] / country_pop[str(year_entering-1)].iloc[0]
        inc_multiplier = country_inc[str(year_entering)].iloc[0] / country_inc[str(year_entering-1)].iloc[0]

        for city_id in city_lookup[country["Number"]]:
            city_data.loc[city_id, "Population"] = math.floor(city_data.loc[city_id, "Population"] * pop_multiplier)
            city_data.loc[city_id, "Income_USDpercap"] *= inc_multiplier

    return city_data
