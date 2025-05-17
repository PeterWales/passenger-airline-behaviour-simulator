import numpy as np
import pandas as pd
import copy
import airline as al
import aircraft as acft
import demand
from constants import (
    OP_HRS_PER_YEAR,
    ROUTE_MAX_SINGLE_SZ,
)


def calc_existing_profits(
    itineraries: pd.DataFrame,
    city_data: pd.DataFrame,
    city_pair_data: pd.DataFrame,
    aircraft_data: pd.DataFrame,
    fleet_data: pd.DataFrame,
    FuelCost_USDperGallon: float,
    demand_coefficients,
) -> pd.DataFrame:
    """
    Calculate profit per seat for all existing routes of an airline.

    Parameters
    ----------
    itineraries : pd.DataFrame
        DataFrame of all itineraries for an airline (element of airline_routes)
    city_data : pd.DataFrame
        DataFrame of city data
    city_pair_data : pd.DataFrame
        DataFrame of city pair data
    aircraft_data : pd.DataFrame
        DataFrame of aircraft data
    fleet_data : pd.DataFrame
        DataFrame of fleet data (element of airline_fleets)
    FuelCost_USDperGallon : float
        Fuel cost in USD per gallon
    
    Returns
    -------
    rtn_flt_df : pd.DataFrame
        DataFrame of all existing routes and their profit per seat for that airline
    """
    already_considered = np.full(len(itineraries), False)
    profit_per_seat_list = []
    origin_list = []
    destination_list = []
    fuel_stop_list = []
    aircraft_id_list = []
    # iterate over all routes operated by that airline
    for row_num, (_, out_itin) in enumerate(itineraries.iterrows()):
        # consider outbound and inbound itineraries together
        if already_considered[row_num]:
            continue

        in_itin_mask = (
            (itineraries["origin"] == out_itin["destination"])
            & (itineraries["destination"] == out_itin["origin"])
            & (itineraries["fuel_stop"] == out_itin["fuel_stop"])
        )
        in_itin_row = itineraries[in_itin_mask].index[0]
        in_itin = itineraries.loc[in_itin_row]
        # Get the row number of the inbound itinerary
        in_itin_row_num = itineraries.index.get_loc(in_itin_row)

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
            itin_seats += aircraft_data.loc[fleet_data.loc[fleet_data["AircraftID"] == ac_id, "SizeClass"].iloc[0], "Seats"]

        # calculate profit per seat of itinerary (outbound + inbound)
        profit_per_seat_list.append(
            (
                al.itin_profit(
                    out_itin,
                    city_pair_out,
                    city_data.loc[out_itin["origin"]],
                    city_data.loc[out_itin["destination"]],
                    fleet_data,
                    aircraft_data,
                    FuelCost_USDperGallon,
                    demand_coefficients,
                ) + al.itin_profit(
                    in_itin,
                    city_pair_in,
                    city_data.loc[out_itin["destination"]],
                    city_data.loc[out_itin["origin"]],
                    fleet_data,
                    aircraft_data,
                    FuelCost_USDperGallon,
                    demand_coefficients,
                )
            ) / itin_seats
        )
        # flag inbound route as already considered
        already_considered[in_itin_row_num] = True

    # create dataframe of all routes and their profit per seat for that airline
    rtn_flt_dict = {
        "Origin": origin_list,
        "Destination": destination_list,
        "Fuel_Stop": fuel_stop_list,
        "Profit_perSeat": profit_per_seat_list,
        "Aircraft_IDs": aircraft_id_list
    }
    rtn_flt_df = pd.DataFrame(rtn_flt_dict)

    return rtn_flt_df


def profit_after_removal(
    itin_ac: pd.DataFrame,
    reassign_ac: pd.Series,
    fleet_df: pd.DataFrame,
    reassign_itin_out: pd.Series,
    reassign_itin_in: pd.Series,
    aircraft_data: pd.DataFrame,
    city_data: pd.DataFrame,
    city_pair_data: pd.DataFrame,
    demand_coefficients: dict[str, float],
    FuelCost_USDperGallon: float,
) -> float:
    """
    Calculate the new profit per seat for an airline itinerary after removing an aircraft to reassign it to a different itinerary.

    Parameters
    ----------
    itin_ac : pd.DataFrame
        DataFrame of all itineraries for an airline (element of airline_routes)
    reassign_ac : pd.Series
        Series of aircraft data for the aircraft being reassigned
    fleet_df : pd.DataFrame
        DataFrame of fleet data (element of airline_fleets)
    reassign_itin_out : pd.Series
        Series of outbound itinerary data for the itinerary being reassigned
    reassign_itin_in : pd.Series
        Series of inbound itinerary data for the itinerary being reassigned
    aircraft_data : pd.DataFrame
        DataFrame of aircraft data
    city_data : pd.DataFrame
        DataFrame of city data
    city_pair_data : pd.DataFrame
        DataFrame of city pair data
    demand_coefficients : dict
        Dictionary of demand coefficients
    FuelCost_USDperGallon : float
        Fuel cost in USD per gallon
    
    Returns
    -------
    reassign_new_profit_per_seat : float
        New profit per seat for the airline itinerary after reassigning the aircraft
    """
    if len(itin_ac) == 1:
        # no aircraft left on this route after reallocation
        reassign_new_profit_per_seat = 0.0
    else:
        # temporarily remove reassign_ac from outbound airline itinerary
        seat_flights_per_year = reassign_ac["Flights_perYear"] * aircraft_data.loc[reassign_ac["SizeClass"], "Seats"]
        reassign_itin_out["flights_per_year"] -= reassign_ac["Flights_perYear"]
        reassign_itin_out["seat_flights_per_year"] -= seat_flights_per_year
        reassign_itin_out["aircraft_ids"].remove(reassign_ac["AircraftID"])
        itin_time_out = acft.calc_itin_time(
            reassign_itin_out,
            city_data,
            city_pair_data,
            aircraft_data,
            fleet_df,
        )
        old_out_exp_utility = reassign_itin_out["exp_utility"]
        reassign_itin_out["exp_utility"] = demand.calc_exp_utility(
            demand_coefficients,
            reassign_itin_out["fare"],
            itin_time_out,
            reassign_itin_out["flights_per_year"],
            reassign_itin_out["fuel_stop"],
        )

        # temporarily remove reassign_ac from inbound airline itinerary
        reassign_itin_in["flights_per_year"] -= reassign_ac["Flights_perYear"]
        reassign_itin_in["seat_flights_per_year"] -= seat_flights_per_year
        reassign_itin_in["aircraft_ids"].remove(reassign_ac["AircraftID"])
        itin_time_in = acft.calc_itin_time(
            reassign_itin_in,
            city_data,
            city_pair_data,
            aircraft_data,
            fleet_df,
        )
        old_in_exp_utility = reassign_itin_in["exp_utility"]
        reassign_itin_in["exp_utility"] = demand.calc_exp_utility(
            demand_coefficients,
            reassign_itin_in["fare"],
            itin_time_in,
            reassign_itin_in["flights_per_year"],
            reassign_itin_in["fuel_stop"],
        )

        # temporarily adjust relevant rows of city_pair_data
        city_pair_out = city_pair_data[
            (city_pair_data["OriginCityID"] == reassign_itin_out["origin"])
            & (city_pair_data["DestinationCityID"] == reassign_itin_out["destination"])
        ].iloc[0]
        city_pair_in = city_pair_data[
            (city_pair_data["OriginCityID"] == reassign_itin_in["origin"])
            & (city_pair_data["DestinationCityID"] == reassign_itin_in["destination"])
        ].iloc[0]
        city_pair_out["Exp_Utility_Sum"] += reassign_itin_out["exp_utility"] - old_out_exp_utility
        city_pair_in["Exp_Utility_Sum"] += reassign_itin_in["exp_utility"] - old_in_exp_utility
        
        old_mean_fare_out = city_pair_out["Mean_Fare_USD"]
        old_mean_fare_in = city_pair_in["Mean_Fare_USD"]
        remaining_seat_flights_per_year = city_pair_out["Seat_Flights_perYear"] - seat_flights_per_year
        if remaining_seat_flights_per_year > 0:
            total_revenue_out = city_pair_out["Mean_Fare_USD"] * city_pair_out["Seat_Flights_perYear"]  # old city pair flights
            total_revenue_in = city_pair_in["Mean_Fare_USD"] * city_pair_in["Seat_Flights_perYear"]
            total_revenue_out -= (reassign_itin_out["fare"] * seat_flights_per_year)  # remove revenue from aircraft being reassigned
            total_revenue_in -= (reassign_itin_in["fare"] * seat_flights_per_year)
            city_pair_out["Mean_Fare_USD"] = total_revenue_out / (remaining_seat_flights_per_year)  # new city pair flights
            city_pair_in["Mean_Fare_USD"] = total_revenue_in / (remaining_seat_flights_per_year)
        else:
            city_pair_out["Mean_Fare_USD"] = city_pair_out["Fare_Est"]
            city_pair_in["Mean_Fare_USD"] = city_pair_in["Fare_Est"]

        # extract relevant cities
        origin = city_data.loc[reassign_itin_out["origin"]]
        destination = city_data.loc[reassign_itin_out["destination"]]

        # calculate sum of all seats that the airline has assigned to this itinerary, minus the aircraft being reassigned
        itin_seats = 0
        for ac_id in reassign_itin_out["aircraft_ids"]:
            itin_seats += aircraft_data.loc[fleet_df.loc[fleet_df["AircraftID"] == ac_id, "SizeClass"].iloc[0], "Seats"]

        reassign_new_profit_per_seat = (
            al.itin_profit(
                reassign_itin_out,
                city_pair_out,
                origin,
                destination,
                fleet_df,
                aircraft_data,
                FuelCost_USDperGallon,
                demand_coefficients,
            ) + al.itin_profit(
                reassign_itin_in,
                city_pair_in,
                destination,
                origin,
                fleet_df,
                aircraft_data,
                FuelCost_USDperGallon,
                demand_coefficients,
            )
        ) / itin_seats

        # reset dataframes to avoid mutability issues
        city_pair_out["Exp_Utility_Sum"] += old_out_exp_utility - reassign_itin_out["exp_utility"]
        city_pair_in["Exp_Utility_Sum"] += old_in_exp_utility - reassign_itin_in["exp_utility"]
        city_pair_out["Mean_Fare_USD"] = old_mean_fare_out
        city_pair_in["Mean_Fare_USD"] = old_mean_fare_in

        reassign_itin_out["flights_per_year"] += reassign_ac["Flights_perYear"]
        reassign_itin_out["seat_flights_per_year"] += seat_flights_per_year
        reassign_itin_out["aircraft_ids"].append(int(reassign_ac["AircraftID"]))
        reassign_itin_out["exp_utility"] = old_out_exp_utility
        reassign_itin_in["flights_per_year"] += reassign_ac["Flights_perYear"]
        reassign_itin_in["seat_flights_per_year"] += seat_flights_per_year
        reassign_itin_in["aircraft_ids"].append(int(reassign_ac["AircraftID"]))
        reassign_itin_in["exp_utility"] = old_in_exp_utility
    
    return reassign_new_profit_per_seat


def find_itin_alternative(
    city_data: pd.DataFrame,
    city_pair_data: pd.DataFrame,
    city_lookup: list[list[int]],
    airline_country: int,
    itineraries: pd.DataFrame,
    fleet_data: pd.DataFrame,
    aircraft_data: pd.DataFrame,
    reassign_ac: pd.Series,
    demand_coefficients: dict[str, float],
    FuelCost_USDperGallon: float,
    reassign_new_profit_per_seat: float,
    reassign_old_profit_per_seat: float,
) -> tuple[float, int, int, int, int]:
    """
    Find an alternative itinerary for an aircraft being reassigned, based on the profit lost by
    removing the aircraft from its current itinerary and the profit gained by adding it to a new itinerary.
    Stop when a beneficial change has been found or all possible alternatives have been tested

    Parameters
    ----------
    city_data : pd.DataFrame
        DataFrame of city data
    city_pair_data : pd.DataFrame
        DataFrame of city pair data
    city_lookup : list
        List of lists of city IDs in each country
    airline_country : int
        ID of the country where the airline is located
    itineraries : pd.DataFrame
        DataFrame of all itineraries for an airline (element of airline_routes)
    fleet_data : pd.DataFrame
        DataFrame of fleet data (element of airline_fleets)
    aircraft_data : pd.DataFrame
        DataFrame of aircraft data
    reassign_ac : pd.Series
        Series of aircraft data for the aircraft being reassigned
    demand_coefficients : dict
        Dictionary of demand coefficients
    FuelCost_USDperGallon : float
        Fuel cost in USD per gallon
    reassign_new_profit_per_seat : float
        New profit per seat for the airline itinerary that the aircraft is being removed from
    reassign_old_profit_per_seat : float
        Old profit per seat for the airline itinerary that the aircraft is being removed from
    
    Returns
    -------
    delta_profit_per_seat : float
        Change in profit per seat for the airline after reassigning the aircraft (sum of impact on new and old itineraries)
    new_origin : int
        ID of the new origin city for the aircraft
    new_destination : int
        ID of the new destination city for the aircraft
    addnl_flights_per_year : int
        Number of flights per year the aircraft can make on the new itinerary
    addnl_seat_flights_per_year : int
        Number of seat flights per year the aircraft can make on the new itinerary
    """
    delta_profit_per_seat = 0.0
    new_origin = -1
    new_destination = -1
    addnl_flights_per_year = 0
    addnl_seat_flights_per_year = 0

    # randomise order of city_pair_data to avoid aircraft always being reassigned to the same routes
    # city_pair_data["RunThis"] column is used to limit simulation to the most popular routes
    random_order = np.random.permutation(city_pair_data[city_pair_data["RunThis"] == 1].index)
    
    for idx in random_order:
        city_pair = city_pair_data.loc[idx].copy()  # copy explicitly to avoid SettingWithCopyWarning
        # don't test the route the aircraft is already on
        if (
            city_pair["OriginCityID"] == reassign_ac["RouteOrigin"]
            and city_pair["DestinationCityID"] == reassign_ac["RouteDestination"]
        ) or (
            city_pair["OriginCityID"] == reassign_ac["RouteDestination"]
            and city_pair["DestinationCityID"] == reassign_ac["RouteOrigin"]
        ):
            continue

        # check whether origin or destination are in country where airline is located
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
                origin_movt_mult = 1.0 + city_data.loc[origin_id, "Movts_Outside_Proportion"]
                destination_movt_mult = 1.0 + city_data.loc[destination_id, "Movts_Outside_Proportion"]
                if (
                    flights_per_year > 0
                    and city_data.loc[origin_id, "Movts_perHr"] + (2.0*origin_movt_mult*flights_per_year/OP_HRS_PER_YEAR) <= city_data.loc[origin_id, "Capacity_MovtsPerHr"]
                    and city_data.loc[destination_id, "Movts_perHr"] + (2.0*destination_movt_mult*flights_per_year/OP_HRS_PER_YEAR) <= city_data.loc[destination_id, "Capacity_MovtsPerHr"]
                ):
                    # new itinerary is possible
                    old_ac_flights_per_year = fleet_data.loc[
                        fleet_data["AircraftID"] == reassign_ac["AircraftID"], "Flights_perYear"
                    ].values[0]

                    seat_flights_per_year = flights_per_year * aircraft_data.loc[reassign_ac["SizeClass"], "Seats"]

                    # check whether airline already flies this route
                    city_pair_in = city_pair_data[
                        (city_pair_data["OriginCityID"] == destination_id)
                        & (city_pair_data["DestinationCityID"] == origin_id)
                    ].iloc[0]
                    if not itineraries[
                        (itineraries["origin"] == origin_id)
                        & (itineraries["destination"] == destination_id)
                        & (itineraries["fuel_stop"] == -1)
                    ].empty:
                        # airline already flies this route

                        # check not exceeding max number of each aircraft size on this route
                        reassign_ac_size = int(reassign_ac["SizeClass"])
                        max_n_same_size = ROUTE_MAX_SINGLE_SZ[f"size_{reassign_ac_size}"]
                        if max_n_same_size != -1:
                            n_same_size = 1  # include the aircraft being reassigned
                            for ac_id in itineraries.loc[
                                (itineraries["origin"] == origin_id)
                                & (itineraries["destination"] == destination_id)
                                & (itineraries["fuel_stop"] == -1), "aircraft_ids"
                            ].iloc[0]:
                                if fleet_data.loc[fleet_data["AircraftID"] == ac_id, "SizeClass"].iloc[0] == reassign_ac_size:
                                    n_same_size += 1
                            if n_same_size > max_n_same_size:
                                continue

                        # calculate old profit per seat of new route without new aircraft
                        out_itin_mask = (
                            (itineraries["origin"] == origin_id)
                            & (itineraries["destination"] == destination_id)
                            & (itineraries["fuel_stop"] == -1)
                        )
                        in_itin_mask = (
                            (itineraries["origin"] == destination_id)
                            & (itineraries["destination"] == origin_id)
                            & (itineraries["fuel_stop"] == -1)
                        )

                        # calculate sum of all seats that the airline has already assigned to this itinerary
                        itin_seats = 0
                        for ac_id in itineraries.loc[out_itin_mask, "aircraft_ids"].iloc[0]:
                            itin_seats += aircraft_data.loc[fleet_data.loc[fleet_data["AircraftID"] == ac_id, "SizeClass"].iloc[0], "Seats"]

                        test_itin_out = copy.deepcopy(itineraries.loc[out_itin_mask])
                        test_itin_in = copy.deepcopy(itineraries.loc[in_itin_mask])

                        new_itin_old_profit_per_seat = (
                            al.itin_profit(
                                test_itin_out.iloc[0],
                                city_pair,
                                city_data.loc[origin_id],
                                city_data.loc[destination_id],
                                fleet_data,
                                aircraft_data,
                                FuelCost_USDperGallon,
                                demand_coefficients,
                            ) + al.itin_profit(
                                test_itin_in.iloc[0],
                                city_pair_in,
                                city_data.loc[destination_id],
                                city_data.loc[origin_id],
                                fleet_data,
                                aircraft_data,
                                FuelCost_USDperGallon,
                                demand_coefficients,
                            )
                        ) / itin_seats

                        # calculate new mean fare for the route after new aircraft assigned
                        old_mean_fare_out = city_pair["Mean_Fare_USD"]
                        old_mean_fare_in = city_pair_in["Mean_Fare_USD"]
                        total_revenue_out = city_pair["Mean_Fare_USD"] * city_pair["Seat_Flights_perYear"]  # old city pair flights
                        total_revenue_in = city_pair_in["Mean_Fare_USD"] * city_pair_in["Seat_Flights_perYear"]
                        total_revenue_out += (test_itin_out["fare"].iloc[0] * seat_flights_per_year)  # add revenue from aircraft being reassigned
                        total_revenue_in += (test_itin_in["fare"].iloc[0] * seat_flights_per_year)
                        city_pair["Mean_Fare_USD"] = total_revenue_out / (city_pair["Seat_Flights_perYear"] + seat_flights_per_year)  # new city pair flights
                        city_pair_in["Mean_Fare_USD"] = total_revenue_in / (city_pair_in["Seat_Flights_perYear"] + seat_flights_per_year)

                        # calculate new profit per seat after new aircraft assigned
                        fleet_data.loc[fleet_data["AircraftID"] == reassign_ac["AircraftID"], "Flights_perYear"] = flights_per_year  # needed for calculating flight cost and time
                        test_itin_out.at[test_itin_out.index[0], "flights_per_year"] += flights_per_year
                        test_itin_out.at[test_itin_out.index[0], "seat_flights_per_year"] += seat_flights_per_year
                        test_itin_out.at[test_itin_out.index[0], "aircraft_ids"].append(int(reassign_ac["AircraftID"]))
                        itin_time_out = acft.calc_itin_time(
                            test_itin_out.iloc[0],
                            city_data,
                            city_pair_data,
                            aircraft_data,
                            fleet_data,
                        )
                        out_old_exp_utility = test_itin_out["exp_utility"].iloc[0]
                        test_itin_out.at[test_itin_out.index[0], "exp_utility"] = demand.calc_exp_utility(
                            demand_coefficients,
                            test_itin_out["fare"].iloc[0],
                            itin_time_out,
                            test_itin_out["flights_per_year"].iloc[0],
                            -1
                        )
                        city_pair["Exp_Utility_Sum"] += test_itin_out["exp_utility"].iloc[0] - out_old_exp_utility

                        test_itin_in.at[test_itin_in.index[0], "flights_per_year"] += flights_per_year
                        test_itin_in.at[test_itin_in.index[0], "seat_flights_per_year"] += seat_flights_per_year
                        test_itin_in.at[test_itin_in.index[0], "aircraft_ids"].append(int(reassign_ac["AircraftID"]))
                        itin_time_in = acft.calc_itin_time(
                            test_itin_in.iloc[0],
                            city_data,
                            city_pair_data,
                            aircraft_data,
                            fleet_data,
                        )
                        in_old_exp_utility = test_itin_in["exp_utility"].iloc[0]
                        test_itin_in.at[test_itin_in.index[0], "exp_utility"] = demand.calc_exp_utility(
                            demand_coefficients,
                            test_itin_in["fare"].iloc[0],
                            itin_time_in,
                            test_itin_in["flights_per_year"].iloc[0],
                            -1
                        )
                        city_pair_in["Exp_Utility_Sum"] += test_itin_in["exp_utility"].iloc[0] - in_old_exp_utility

                        # add reassigned aircraft to sum of seats calculated earlier
                        itin_seats += aircraft_data.loc[reassign_ac["SizeClass"], "Seats"]

                        new_itin_new_profit_per_seat = (
                            al.itin_profit(
                                test_itin_out.iloc[0],
                                city_pair,
                                city_data.loc[origin_id],
                                city_data.loc[destination_id],
                                fleet_data,
                                aircraft_data,
                                FuelCost_USDperGallon,
                                demand_coefficients,
                            ) + al.itin_profit(
                                test_itin_in.iloc[0],
                                city_pair_in,
                                city_data.loc[destination_id],
                                city_data.loc[origin_id],
                                fleet_data,
                                aircraft_data,
                                FuelCost_USDperGallon,
                                demand_coefficients,
                            )
                        ) / itin_seats

                        # reset dataframes to avoid mutability issues
                        city_pair["Mean_Fare_USD"] = old_mean_fare_out
                        city_pair_in["Mean_Fare_USD"] = old_mean_fare_in
                        city_pair["Exp_Utility_Sum"] += out_old_exp_utility - test_itin_out["exp_utility"].iloc[0]
                        city_pair_in["Exp_Utility_Sum"] += in_old_exp_utility - test_itin_in["exp_utility"].iloc[0]
                        test_itin_out.at[test_itin_out.index[0], "flights_per_year"] -= flights_per_year
                        test_itin_out.at[test_itin_out.index[0], "seat_flights_per_year"] -= seat_flights_per_year
                        test_itin_out.at[test_itin_out.index[0], "aircraft_ids"].remove(reassign_ac["AircraftID"])
                        test_itin_out.at[test_itin_out.index[0], "exp_utility"] = out_old_exp_utility
                        test_itin_in.at[test_itin_in.index[0], "flights_per_year"] -= flights_per_year
                        test_itin_in.at[test_itin_in.index[0], "seat_flights_per_year"] -= seat_flights_per_year
                        test_itin_in.at[test_itin_in.index[0], "aircraft_ids"].remove(reassign_ac["AircraftID"])
                        test_itin_in.at[test_itin_in.index[0], "exp_utility"] = in_old_exp_utility
                        fleet_data.loc[fleet_data["AircraftID"] == reassign_ac["AircraftID"], "Flights_perYear"] = old_ac_flights_per_year
                    else:
                        # airline doesn't already fly this route
                        new_itin_old_profit_per_seat = 0.0
                        fleet_data.loc[fleet_data["AircraftID"] == reassign_ac["AircraftID"], "Flights_perYear"] = flights_per_year  # needed for calculating flight cost and time
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
                        
                        test_itin_time_out = acft.calc_itin_time(
                            test_itin_out,
                            city_data,
                            city_pair_data,
                            aircraft_data,
                            fleet_data,
                        )
                        test_itin_time_in = acft.calc_itin_time(
                            test_itin_in,
                            city_data,
                            city_pair_data,
                            aircraft_data,
                            fleet_data,
                        )
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
                        city_pair["Exp_Utility_Sum"] += test_itin_out_exp_utility
                        city_pair_in["Exp_Utility_Sum"] += test_itin_in_exp_utility

                        # note route mean fare doesn't need to be adjusted here because new itinerary is initialised with the mean fare

                        # calculate profit per seat of new itinerary (outbound + inbound)
                        new_itin_new_profit_per_seat = (
                            al.itin_profit(
                                test_itin_out,
                                city_pair,
                                city_data.loc[origin_id],
                                city_data.loc[destination_id],
                                fleet_data,
                                aircraft_data,
                                FuelCost_USDperGallon,
                                demand_coefficients,
                            ) + al.itin_profit(
                                test_itin_in,
                                city_pair_in,
                                city_data.loc[destination_id],
                                city_data.loc[origin_id],
                                fleet_data,
                                aircraft_data,
                                FuelCost_USDperGallon,
                                demand_coefficients,
                            )
                        ) / aircraft_data.loc[reassign_ac["SizeClass"], "Seats"]  # seats on this itinerary are only provided by reassigned aircraft

                        # reset dataframes to avoid mutability issues
                        fleet_data.loc[fleet_data["AircraftID"] == reassign_ac["AircraftID"], "Flights_perYear"] = old_ac_flights_per_year
                        city_pair["Exp_Utility_Sum"] -= test_itin_out_exp_utility
                        city_pair_in["Exp_Utility_Sum"] -= test_itin_in_exp_utility
                        # note test_itin_out and test_itin_in are never added to airline_routes dataframe, so no need to reset them

                    # check new itinerary is actually profitable
                    if new_itin_new_profit_per_seat > 0:
                        # calculate change in profit per seat
                        test_delta_profit_per_seat = (
                            new_itin_new_profit_per_seat - new_itin_old_profit_per_seat
                            + reassign_new_profit_per_seat - reassign_old_profit_per_seat
                        )

                        # if test makes more profit than current itinerary, save it and stop searching
                        if test_delta_profit_per_seat > 0:
                            delta_profit_per_seat = test_delta_profit_per_seat
                            new_origin = origin_id
                            new_destination = destination_id
                            addnl_flights_per_year = flights_per_year
                            addnl_seat_flights_per_year = seat_flights_per_year
                            break
    
    return delta_profit_per_seat, new_origin, new_destination, addnl_flights_per_year, addnl_seat_flights_per_year


def reassign_ac_to_new_route(
    new_origin: int,
    new_destination: int,
    out_reassign_mask: pd.Series | None,
    in_reassign_mask: pd.Series | None,
    reassign_itin_out: pd.Series | None,
    reassign_itin_in: pd.Series | None,
    reassign_ac: pd.Series,
    addnl_flights_per_year: int,
    addnl_seat_flights_per_year: int,
    city_pair_data: pd.DataFrame,
    city_data: pd.DataFrame,
    airline_routes: pd.DataFrame,
    airlines: pd.DataFrame,
    airline_id: int,
    airline_fleets: pd.DataFrame,
    aircraft_data: pd.DataFrame,
    demand_coefficients: dict[str, float],
):
    """
    Reassign an aircraft to a new route (or ground it) and make the necessary adjustments to all dataframes.

    Parameters
    ----------
    new_origin : int
        ID of the new origin city for the aircraft, or -1 if aircraft is being grounded
    new_destination : int
        ID of the new destination city for the aircraft, or -1 if aircraft is being grounded
    out_reassign_mask : pd.Series
        Mask for the outbound itinerary the aircraft is being removed from, or None if aircraft was previously grounded
    in_reassign_mask : pd.Series
        Mask for the inbound itinerary the aircraft is being removed from, or None if aircraft was previously grounded
    reassign_itin_out : pd.Series
        Series of outbound itinerary data for the itinerary the aircraft is being removed from, or None if aircraft was previously grounded
    reassign_itin_in : pd.Series
        Series of inbound itinerary data for the itinerary the aircraft is being removed from, or None if aircraft was previously grounded
    reassign_ac : pd.Series
        Series of aircraft data for the aircraft being reassigned
    addnl_flights_per_year : int
        Number of flights per year the aircraft can make on the new itinerary
    addnl_seat_flights_per_year : int
        Number of seat flights per year the aircraft can make on the new itinerary
    city_pair_data : pd.DataFrame
        DataFrame of city pair data
    city_data : pd.DataFrame
        DataFrame of city data
    airline_routes : pd.DataFrame
        DataFrame of all itineraries for an airline (element of airline_routes)
    airlines : pd.DataFrame
        DataFrame of airline data
    airline_id : int
        ID of the airline
    airline_fleets : pd.DataFrame
        DataFrame of fleet data (element of airline_fleets)
    aircraft_data : pd.DataFrame
        DataFrame of aircraft data
    demand_coefficients : dict
        Dictionary of demand coefficients
    
    Returns
    -------
    airlines : pd.DataFrame
    airline_routes : pd.DataFrame
    airline_fleets : pd.DataFrame
    city_data : pd.DataFrame
    city_pair_data : pd.DataFrame
    """
    if new_origin == -1:
        grounding = True  # aircraft is being moved from an itinerary to storage
        deploying = False
    else:
        grounding = False
        if out_reassign_mask is None:
            deploying = True  # aircraft is being assigned to a new itinerary from storage or new lease
        else:
            deploying = False

    if not grounding:
        # routes that the aircraft is being added to
        new_city_pair_out = city_pair_data[
            (city_pair_data["OriginCityID"] == new_origin)
            & (city_pair_data["DestinationCityID"] == new_destination)
        ].iloc[0]
        new_city_pair_in = city_pair_data[
            (city_pair_data["OriginCityID"] == new_destination)
            & (city_pair_data["DestinationCityID"] == new_origin)
        ].iloc[0]

        new_route_revenue_out = new_city_pair_out["Mean_Fare_USD"] * new_city_pair_out["Seat_Flights_perYear"]
        new_route_revenue_in = new_city_pair_in["Mean_Fare_USD"] * new_city_pair_in["Seat_Flights_perYear"]

        # adjust new city_pair_data Seat_Flights_perYear
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
    
    if not deploying:
        old_city_pair_out = city_pair_data[
            (city_pair_data["OriginCityID"] == reassign_itin_out["origin"])
            & (city_pair_data["DestinationCityID"] == reassign_itin_out["destination"])
        ].iloc[0]
        old_city_pair_in = city_pair_data[
            (city_pair_data["OriginCityID"] == reassign_itin_in["origin"])
            & (city_pair_data["DestinationCityID"] == reassign_itin_in["destination"])
        ].iloc[0]

        old_route_revenue_out = old_city_pair_out["Mean_Fare_USD"] * old_city_pair_out["Seat_Flights_perYear"]
        old_route_revenue_in = old_city_pair_in["Mean_Fare_USD"] * old_city_pair_in["Seat_Flights_perYear"]

        # save old exp(utility) for calculating deltas for city_pair_data
        reassign_old_utility_out = reassign_itin_out["exp_utility"]
        reassign_old_utility_in = reassign_itin_in["exp_utility"]

        # adjust old city_pair_data Seat_Flights_perYear
        city_pair_data.loc[
            (city_pair_data["OriginCityID"] == reassign_itin_out["origin"])
            & (city_pair_data["DestinationCityID"] == reassign_itin_out["destination"]),
            "Seat_Flights_perYear"
        ] -= reassign_ac["Flights_perYear"] * aircraft_data.loc[reassign_ac["SizeClass"], "Seats"]
        city_pair_data.loc[
            (city_pair_data["OriginCityID"] == reassign_itin_in["origin"])
            & (city_pair_data["DestinationCityID"] == reassign_itin_in["destination"]),
            "Seat_Flights_perYear"
        ] -= reassign_ac["Flights_perYear"] * aircraft_data.loc[reassign_ac["SizeClass"], "Seats"]

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

    if not deploying:
        # remove aircraft and associated data from old airline itinerary
        airline_routes[airline_id].loc[out_reassign_mask, "aircraft_ids"].iloc[0].remove(reassign_ac["AircraftID"])
        airline_routes[airline_id].loc[in_reassign_mask, "aircraft_ids"].iloc[0].remove(reassign_ac["AircraftID"])
        airline_routes[airline_id].loc[out_reassign_mask, "flights_per_year"] -= reassign_ac["Flights_perYear"]
        airline_routes[airline_id].loc[in_reassign_mask, "flights_per_year"] -= reassign_ac["Flights_perYear"]
        airline_routes[airline_id].loc[out_reassign_mask, "seat_flights_per_year"] -= reassign_ac["Flights_perYear"] * aircraft_data.loc[reassign_ac["SizeClass"], "Seats"]
        airline_routes[airline_id].loc[in_reassign_mask, "seat_flights_per_year"] -= reassign_ac["Flights_perYear"] * aircraft_data.loc[reassign_ac["SizeClass"], "Seats"]
        # check whether there are still any aircraft assigned to this itinerary
        if len(airline_routes[airline_id].loc[out_reassign_mask, "aircraft_ids"].iloc[0]) > 0:
            reassign_itin_time_out = acft.calc_itin_time(
                airline_routes[airline_id].loc[out_reassign_mask].iloc[0],
                city_data,
                city_pair_data,
                aircraft_data,
                airline_fleets[airline_id],
            )
            reassign_itin_time_in = acft.calc_itin_time(
                airline_routes[airline_id].loc[in_reassign_mask].iloc[0],
                city_data,
                city_pair_data,
                aircraft_data,
                airline_fleets[airline_id],
            )
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

            # update city_pair_data mean fare for old route
            subtract_seat_flights_per_year = reassign_ac["Flights_perYear"] * aircraft_data.loc[reassign_ac["SizeClass"], "Seats"]
            subtract_revenue_out = subtract_seat_flights_per_year * reassign_itin_out["fare"]
            subtract_revenue_in = reassign_ac["Flights_perYear"] * aircraft_data.loc[reassign_ac["SizeClass"], "Seats"] * reassign_itin_in["fare"]
            total_revenue_out = old_route_revenue_out - subtract_revenue_out
            total_revenue_in = old_route_revenue_in - subtract_revenue_in
            city_pair_data.loc[
                (city_pair_data["OriginCityID"] == reassign_itin_out["origin"])
                & (city_pair_data["DestinationCityID"] == reassign_itin_out["destination"]),
                "Mean_Fare_USD"
            ] = total_revenue_out / (old_city_pair_out["Seat_Flights_perYear"] - subtract_seat_flights_per_year)
            city_pair_data.loc[
                (city_pair_data["OriginCityID"] == reassign_itin_in["origin"])
                & (city_pair_data["DestinationCityID"] == reassign_itin_in["destination"]),
                "Mean_Fare_USD"
            ] = total_revenue_in / (old_city_pair_in["Seat_Flights_perYear"] - subtract_seat_flights_per_year)
        else:
            airline_routes[airline_id].loc[out_reassign_mask, "exp_utility"] = 0.0
            airline_routes[airline_id].loc[in_reassign_mask, "exp_utility"] = 0.0

            # update city_pair_data mean fare for old route
            city_pair_data.loc[
                (city_pair_data["OriginCityID"] == reassign_itin_out["origin"])
                & (city_pair_data["DestinationCityID"] == reassign_itin_out["destination"]),
                "Mean_Fare_USD"
            ] = city_pair_data.loc[
                (city_pair_data["OriginCityID"] == reassign_itin_out["origin"])
                & (city_pair_data["DestinationCityID"] == reassign_itin_out["destination"]),
                "Fare_Est"
            ].iloc[0]
            city_pair_data.loc[
                (city_pair_data["OriginCityID"] == reassign_itin_in["origin"])
                & (city_pair_data["DestinationCityID"] == reassign_itin_in["destination"]),
                "Mean_Fare_USD"
            ] = city_pair_data.loc[
                (city_pair_data["OriginCityID"] == reassign_itin_in["origin"])
                & (city_pair_data["DestinationCityID"] == reassign_itin_in["destination"]),
                "Fare_Est"
            ].iloc[0]

        # calculate exp(utility) deltas for city_pair_data
        reassign_delta_exp_utility_out = airline_routes[airline_id].loc[out_reassign_mask, "exp_utility"].iloc[0] - reassign_old_utility_out
        reassign_delta_exp_utility_in = airline_routes[airline_id].loc[in_reassign_mask, "exp_utility"].iloc[0] - reassign_old_utility_in
    else:
        # remove aircraft from grounded_acft
        airlines.loc[airline_id, "Grounded_acft"].remove(reassign_ac["AircraftID"])

    if not grounding:
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
            new_itin_time_out = acft.calc_itin_time(
                airline_routes[airline_id].loc[out_itin_mask].iloc[0],
                city_data,
                city_pair_data,
                aircraft_data,
                airline_fleets[airline_id],
            )
            new_itin_time_in = acft.calc_itin_time(
                airline_routes[airline_id].loc[in_itin_mask].iloc[0],
                city_data,
                city_pair_data,
                aircraft_data,
                airline_fleets[airline_id],
            )
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

            # update city_pair_data mean fare for new route
            addnl_revenue_out = addnl_seat_flights_per_year * airline_routes[airline_id].loc[out_itin_mask, "fare"].iloc[0]
            addnl_revenue_in = addnl_seat_flights_per_year * airline_routes[airline_id].loc[in_itin_mask, "fare"].iloc[0]
            total_revenue_out = new_route_revenue_out + addnl_revenue_out
            total_revenue_in = new_route_revenue_in + addnl_revenue_in
            city_pair_data.loc[
                (city_pair_data["OriginCityID"] == new_origin)
                & (city_pair_data["DestinationCityID"] == new_destination),
                "Mean_Fare_USD"
            ] = total_revenue_out / city_pair_data.loc[
                (city_pair_data["OriginCityID"] == new_origin)
                & (city_pair_data["DestinationCityID"] == new_destination),
                "Seat_Flights_perYear"
            ]
            city_pair_data.loc[
                (city_pair_data["OriginCityID"] == new_destination)
                & (city_pair_data["DestinationCityID"] == new_origin),
                "Mean_Fare_USD"
            ] = total_revenue_in / city_pair_data.loc[
                (city_pair_data["OriginCityID"] == new_destination)
                & (city_pair_data["DestinationCityID"] == new_origin),
                "Seat_Flights_perYear"
            ]

        else:
            # airline doesn't already fly this route
            new_itin_out = {
                "origin": new_origin,
                "destination": new_destination,
                "fare": new_city_pair_out["Mean_Fare_USD"],
                "aircraft_ids": [reassign_ac["AircraftID"]],
                "flights_per_year": addnl_flights_per_year,
                "seat_flights_per_year": addnl_seat_flights_per_year,
                "exp_utility": 0.0,  # calculated later
                "fuel_stop": -1,
                "itin_time_hrs": 0.0,  # calculated later
            }
            new_itin_in = {
                "origin": new_destination,
                "destination": new_origin,
                "fare": new_city_pair_in["Mean_Fare_USD"],
                "aircraft_ids": [reassign_ac["AircraftID"]],
                "flights_per_year": addnl_flights_per_year,
                "seat_flights_per_year": addnl_seat_flights_per_year,
                "exp_utility": 0.0,  # calculated later
                "fuel_stop": -1,
                "itin_time_hrs": 0.0,  # calculated later
            }
            new_itin_out = pd.Series(new_itin_out)
            new_itin_in = pd.Series(new_itin_in)

            new_itin_time_out = acft.calc_itin_time(
                new_itin_out,
                city_data,
                city_pair_data,
                aircraft_data,
                airline_fleets[airline_id],
            )
            new_itin_time_in = acft.calc_itin_time(
                new_itin_in,
                city_data,
                city_pair_data,
                aircraft_data,
                airline_fleets[airline_id],
            )
            new_itin_out["itin_time_hrs"] = new_itin_time_out
            new_itin_in["itin_time_hrs"] = new_itin_time_in

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

            # note mean fare doesn't need to be adjusted here because new itinerary is initialised with the mean fare
    else:
        # add aircraft to grounded_acft
        airlines.loc[airline_id, "Grounded_acft"].append(reassign_ac["AircraftID"])

    if not deploying:
        # adjust city_data movements
        origin_movt_mult = 1.0 + city_data.loc[reassign_itin_out["origin"], "Movts_Outside_Proportion"]
        destination_movt_mult = 1.0 + city_data.loc[reassign_itin_out["destination"], "Movts_Outside_Proportion"]
        city_data.loc[reassign_itin_out["origin"], "Movts_perHr"] -= 2.0 * origin_movt_mult * reassign_ac["Flights_perYear"]/OP_HRS_PER_YEAR
        city_data.loc[reassign_itin_out["destination"], "Movts_perHr"] -= 2.0 * destination_movt_mult * reassign_ac["Flights_perYear"]/OP_HRS_PER_YEAR
        if reassign_itin_out["fuel_stop"] != -1:
            fuel_stop_movt_mult = 1.0 + city_data.loc[reassign_itin_out["fuel_stop"], "Movts_Outside_Proportion"]
            city_data.loc[reassign_itin_out["fuel_stop"], "Movts_perHr"] -= 4.0 * fuel_stop_movt_mult * reassign_ac["Flights_perYear"]/OP_HRS_PER_YEAR
        
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
    
    if not grounding:
        # adjust city_data movements
        origin_movt_mult = 1.0 + city_data.loc[new_origin, "Movts_Outside_Proportion"]
        destination_movt_mult = 1.0 + city_data.loc[new_destination, "Movts_Outside_Proportion"]
        city_data.loc[new_origin, "Movts_perHr"] += 2.0 * origin_movt_mult * addnl_flights_per_year/OP_HRS_PER_YEAR
        city_data.loc[new_destination, "Movts_perHr"] += 2.0 * destination_movt_mult * addnl_flights_per_year/OP_HRS_PER_YEAR

        # adjust city_pair_data Exp_Utility_Sum
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

    return airlines, airline_routes, airline_fleets, city_data, city_pair_data


def update_profit_tracker(
    rtn_flt_df: pd.DataFrame,
    new_origin: int,
    new_destination: int,
    out_reassign_mask: pd.Series,
    in_reassign_mask: pd.Series,
    reassign_itin_out: pd.Series,
    reassign_ac: pd.Series,
    airline_routes: pd.DataFrame,
    airline_fleets: pd.DataFrame,
    airline_id: int,
    city_pair_data: pd.DataFrame,
    city_data: pd.DataFrame,
    aircraft_data: pd.DataFrame,
    FuelCost_USDperGallon: float,
    demand_coefficients: dict[str, float],
) -> pd.DataFrame:
    """
    Update rtn_flt_df with new profit per seat and aircraft IDs after reassigning an aircraft to a different itinerary.

    Parameters
    ----------
    rtn_flt_df : pd.DataFrame
        DataFrame of all itineraries for an airline and their profit per seat and aircraft IDs
    new_origin : int
        ID of the new origin city for the aircraft, or -1 if aircraft is being grounded
    new_destination : int
        ID of the new destination city for the aircraft, or -1 if aircraft is being grounded
    out_reassign_mask : pd.Series
        Mask for the outbound itinerary the aircraft is being removed from
    in_reassign_mask : pd.Series
        Mask for the inbound itinerary the aircraft is being removed from
    reassign_itin_out : pd.Series
        Series of outbound itinerary data for the itinerary the aircraft is being removed from
    reassign_ac : pd.Series
        Series of aircraft data for the aircraft being reassigned
    airline_routes : pd.DataFrame
        DataFrame of all itineraries for an airline (element of airline_routes)
    airline_fleets : pd.DataFrame
        DataFrame of fleet data (element of airline_fleets)
    airline_id : int
        ID of the airline
    city_pair_data : pd.DataFrame
        DataFrame of city pair data
    city_data : pd.DataFrame
        DataFrame of city data
    aircraft_data : pd.DataFrame
        DataFrame of aircraft data
    FuelCost_USDperGallon : float
        Cost of fuel in USD per gallon
    demand_coefficients : dict
        Dictionary of demand coefficients
    
    Returns
    -------
    rtn_flt_df : pd.DataFrame
    """
    if new_origin == -1:
        grounding = True  # aircraft is being moved from an itinerary to storage
        deploying = False
    else:
        grounding = False
        if out_reassign_mask is None:
            deploying = True  # aircraft is being assigned to a new itinerary from storage or new lease
        else:
            deploying = False

    if not grounding:
        # check if new itinerary already exists in rtn_flt_df
        if (
            rtn_flt_df[
                (rtn_flt_df["Origin"] == new_origin)
                & (rtn_flt_df["Destination"] == new_destination)
                & (rtn_flt_df["Fuel_Stop"] == -1)
            ].empty
            and rtn_flt_df[
                (rtn_flt_df["Origin"] == new_destination)
                & (rtn_flt_df["Destination"] == new_origin)
                & (rtn_flt_df["Fuel_Stop"] == -1)
            ].empty
        ):
            # add itinerary to rtn_flt_df
            new_itin_dict = {
                "Origin": [new_origin],
                "Destination": [new_destination],
                "Fuel_Stop": [-1],
                "Profit_perSeat": [0],  # recalculated later
                "Aircraft_IDs": [[reassign_ac["AircraftID"]]]
            }
            new_itin_df = pd.DataFrame(new_itin_dict)
            rtn_flt_df = pd.concat([rtn_flt_df, new_itin_df], ignore_index=True)

    reassign_mask = (
        (rtn_flt_df["Origin"] == reassign_itin_out["origin"])
        & (rtn_flt_df["Destination"] == reassign_itin_out["destination"])
        & (rtn_flt_df["Fuel_Stop"] == reassign_itin_out["fuel_stop"])
    )
    if len(rtn_flt_df.loc[reassign_mask]) == 0:
        reassign_mask = (
            (rtn_flt_df["Origin"] == reassign_itin_out["destination"])
            & (rtn_flt_df["Destination"] == reassign_itin_out["origin"])
            & (rtn_flt_df["Fuel_Stop"] == reassign_itin_out["fuel_stop"])
        )

    # recalculate profit per seat of altered itineraries
    if not deploying:
        # itinerary aircraft is being removed from
        updated_reassign_itin_out = airline_routes[airline_id].loc[out_reassign_mask]
        updated_reassign_itin_in = airline_routes[airline_id].loc[in_reassign_mask]
        updated_reassign_city_pair_out = city_pair_data.loc[
            (city_pair_data["OriginCityID"] == reassign_itin_out["origin"])
            & (city_pair_data["DestinationCityID"] == reassign_itin_out["destination"])
        ].iloc[0]
        updated_reassign_city_pair_in = city_pair_data.loc[
            (city_pair_data["OriginCityID"] == reassign_itin_out["destination"])
            & (city_pair_data["DestinationCityID"] == reassign_itin_out["origin"])
        ].iloc[0]
        # calculate sum of all seats that the airline has assigned to this itinerary minus reassigned aircraft
        itin_seats = 0
        for ac_id in updated_reassign_itin_out["aircraft_ids"].iloc[0]:
            itin_seats += aircraft_data.loc[airline_fleets[airline_id].loc[airline_fleets[airline_id]["AircraftID"] == ac_id, "SizeClass"].iloc[0], "Seats"]
        if itin_seats == 0:
            rtn_flt_df.loc[reassign_mask, "Profit_perSeat"] = 0.0
        else:
            rtn_flt_df.loc[reassign_mask, "Profit_perSeat"] = (
                al.itin_profit(
                    updated_reassign_itin_out.iloc[0],
                    updated_reassign_city_pair_out,
                    city_data.loc[reassign_itin_out["origin"]],
                    city_data.loc[reassign_itin_out["destination"]],
                    airline_fleets[airline_id],
                    aircraft_data,
                    FuelCost_USDperGallon,
                    demand_coefficients,
                ) + al.itin_profit(
                    updated_reassign_itin_in.iloc[0],
                    updated_reassign_city_pair_in,
                    city_data.loc[reassign_itin_out["destination"]],
                    city_data.loc[reassign_itin_out["origin"]],
                    airline_fleets[airline_id],
                    aircraft_data,
                    FuelCost_USDperGallon,
                    demand_coefficients,
                )
            ) / itin_seats

    new_itin_mask = (
        (rtn_flt_df["Origin"] == new_origin)
        & (rtn_flt_df["Destination"] == new_destination)
        & (rtn_flt_df["Fuel_Stop"] == -1)
    )
    if len(rtn_flt_df.loc[new_itin_mask]) == 0:
        new_itin_mask = (
            (rtn_flt_df["Origin"] == new_destination)
            & (rtn_flt_df["Destination"] == new_origin)
            & (rtn_flt_df["Fuel_Stop"] == -1)
        )

    if not grounding:
        # itinerary aircraft is being added to
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
        updated_new_city_pair_out = city_pair_data.loc[
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
            itin_seats += aircraft_data.loc[airline_fleets[airline_id].loc[airline_fleets[airline_id]["AircraftID"] == ac_id, "SizeClass"].iloc[0], "Seats"]
        rtn_flt_df.loc[new_itin_mask, "Profit_perSeat"] = (
            al.itin_profit(
                updated_new_itin_out.iloc[0],
                updated_new_city_pair_out,
                city_data.loc[new_origin],
                city_data.loc[new_destination],
                airline_fleets[airline_id],
                aircraft_data,
                FuelCost_USDperGallon,
                demand_coefficients,
            ) + al.itin_profit(
                updated_new_itin_in.iloc[0],
                updated_city_pair_in,
                city_data.loc[new_destination],
                city_data.loc[new_origin],
                airline_fleets[airline_id],
                aircraft_data,
                FuelCost_USDperGallon,
                demand_coefficients,
            )
        ) / itin_seats

    # move reassigned aircraft in rtn_flt_df
    if not deploying:
        rtn_flt_df.loc[
            reassign_mask, "Aircraft_IDs"
        ].iloc[0].remove(reassign_ac["AircraftID"])
    if not grounding:
        # aircraft will already exist in new itinerary if it is new to the airline
        if reassign_ac["AircraftID"] not in rtn_flt_df.loc[new_itin_mask, "Aircraft_IDs"].iloc[0]:
            rtn_flt_df.loc[new_itin_mask, "Aircraft_IDs"].iloc[0].append(reassign_ac["AircraftID"])
    
    return rtn_flt_df
