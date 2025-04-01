import numpy as np
import pandas as pd
import copy
import airline as al
import aircraft as acft
import demand


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
    for row_num, (out_itin_idx, out_itin) in enumerate(itineraries.iterrows()):
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
            itin_seats += aircraft_data.loc[fleet_data.loc[ac_id, "SizeClass"], "Seats"]

        # calculate profit per seat of itinerary (outbound + inbound)
        profit_per_seat_list.append(
            (
                al.itin_profit(
                    out_itin["fare"],
                    out_itin,
                    city_pair_out,
                    city_data.loc[out_itin["origin"]],
                    city_data.loc[out_itin["destination"]],
                    fleet_data,
                    aircraft_data,
                    FuelCost_USDperGallon,
                    demand_coefficients,
                ) + al.itin_profit(
                    in_itin["fare"],
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
        # remove reassign_ac from airline itinerary
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
        old_in_exp_utility = reassign_itin_in["exp_utility"]
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
            itin_seats += aircraft_data.loc[fleet_df.loc[ac_id, "SizeClass"], "Seats"]

        reassign_new_profit_per_seat = (
            al.itin_profit(
                reassign_itin_out["fare"],
                reassign_itin_out,
                city_pair,
                origin,
                destination,
                fleet_df,
                aircraft_data,
                FuelCost_USDperGallon,
                demand_coefficients,
                add_city_pair_seat_flights = -seat_flights_per_year,
                add_city_pair_exp_utility = reassign_itin_out["exp_utility"] - old_out_exp_utility,
            ) + al.itin_profit(
                reassign_itin_in["fare"],
                reassign_itin_in,
                city_pair,
                destination,
                origin,
                fleet_df,
                aircraft_data,
                FuelCost_USDperGallon,
                demand_coefficients,
                add_city_pair_seat_flights = -seat_flights_per_year,
                add_city_pair_exp_utility = reassign_itin_in["exp_utility"] - old_in_exp_utility,
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
    
    return reassign_new_profit_per_seat


def best_itin_alternative(
    city_data: pd.DataFrame,
    city_pair_data: pd.DataFrame,
    city_lookup: list[list[int]],
    airline_country: int,
    itineraries: pd.DataFrame,
    fleet_data: pd.DataFrame,
    aircraft_data: pd.DataFrame,
    reassign_ac: pd.Series,
    op_hrs_per_year: float,
    demand_coefficients: dict[str, float],
    FuelCost_USDperGallon: float,
    reassign_new_profit_per_seat: float,
    reassign_old_profit_per_seat: float,
):
    """
    Find the best alternative itinerary for an aircraft being reassigned, based on the profit lost by
    removing the aircraft from its current itinerary and the profit gained by adding it to a new itinerary.

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
    op_hrs_per_year : float
        Number of operating hours per year
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

    for _, city_pair in city_pair_data.iterrows():
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
                if (
                    flights_per_year > 0
                    and city_data.loc[origin_id, "Movts_perHr"] + (2*float(flights_per_year)/op_hrs_per_year) <= city_data.loc[origin_id, "Capacity_MovtsPerHr"]
                    and city_data.loc[destination_id, "Movts_perHr"] + (2*float(flights_per_year)/op_hrs_per_year) <= city_data.loc[destination_id, "Capacity_MovtsPerHr"]
                ):
                    # new itinerary is possible
                    old_ac_flights_per_year = fleet_data.loc[reassign_ac["AircraftID"], "Flights_perYear"]

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
                            itin_seats += aircraft_data.loc[fleet_data.loc[ac_id, "SizeClass"], "Seats"]

                        test_itin_out = copy.deepcopy(itineraries.loc[out_itin_mask])
                        test_itin_in = copy.deepcopy(itineraries.loc[in_itin_mask])

                        new_itin_old_profit_per_seat = (
                            al.itin_profit(
                                test_itin_out["fare"].iloc[0],
                                test_itin_out.iloc[0],
                                city_pair,
                                city_data.loc[origin_id],
                                city_data.loc[destination_id],
                                fleet_data,
                                aircraft_data,
                                FuelCost_USDperGallon,
                                demand_coefficients,
                            ) + al.itin_profit(
                                test_itin_in["fare"].iloc[0],
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

                        # calculate new profit per seat after new aircraft assigned
                        fleet_data.loc[reassign_ac["AircraftID"], "Flights_perYear"] = flights_per_year  # needed for calculating flight cost
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

                        # calculate sum of all seats that the airline has assigned to this itinerary including reassigned aircraft
                        itin_seats += aircraft_data.loc[reassign_ac["SizeClass"], "Seats"]

                        new_itin_new_profit_per_seat = (
                            al.itin_profit(
                                test_itin_out["fare"].iloc[0],
                                test_itin_out.iloc[0],
                                city_pair,
                                city_data.loc[origin_id],
                                city_data.loc[destination_id],
                                fleet_data,
                                aircraft_data,
                                FuelCost_USDperGallon,
                                demand_coefficients,
                                add_city_pair_seat_flights=seat_flights_per_year,
                                add_city_pair_exp_utility=(
                                    test_itin_out["exp_utility"].iloc[0]
                                    - out_old_exp_utility
                                )
                            ) + al.itin_profit(
                                test_itin_in["fare"].iloc[0],
                                test_itin_in.iloc[0],
                                city_pair_in,
                                city_data.loc[destination_id],
                                city_data.loc[origin_id],
                                fleet_data,
                                aircraft_data,
                                FuelCost_USDperGallon,
                                demand_coefficients,
                                add_city_pair_seat_flights=seat_flights_per_year,
                                add_city_pair_exp_utility=(
                                    test_itin_in["exp_utility"].iloc[0]
                                    - in_old_exp_utility
                                )
                            )
                        ) / itin_seats

                        # reset test_itin_out, test_itin_in and fleet_data to avoid mutability issues
                        test_itin_out.at[test_itin_out.index[0], "flights_per_year"] -= flights_per_year
                        test_itin_out.at[test_itin_out.index[0], "seat_flights_per_year"] -= seat_flights_per_year
                        test_itin_out.at[test_itin_out.index[0], "aircraft_ids"].remove(reassign_ac["AircraftID"])
                        test_itin_out.at[test_itin_out.index[0], "exp_utility"] = out_old_exp_utility
                        test_itin_in.at[test_itin_in.index[0], "flights_per_year"] -= flights_per_year
                        test_itin_in.at[test_itin_in.index[0], "seat_flights_per_year"] -= seat_flights_per_year
                        test_itin_in.at[test_itin_in.index[0], "aircraft_ids"].remove(reassign_ac["AircraftID"])
                        test_itin_in.at[test_itin_in.index[0], "exp_utility"] = in_old_exp_utility
                        fleet_data.loc[reassign_ac["AircraftID"], "Flights_perYear"] = old_ac_flights_per_year
                    else:
                        # airline doesn't already fly this route
                        new_itin_old_profit_per_seat = 0.0
                        fleet_data.loc[reassign_ac["AircraftID"], "Flights_perYear"] = flights_per_year  # needed for calculating flight cost
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
                            al.itin_profit(
                                test_itin_out["fare"],
                                test_itin_out,
                                city_pair,
                                city_data.loc[origin_id],
                                city_data.loc[destination_id],
                                fleet_data,
                                aircraft_data,
                                FuelCost_USDperGallon,
                                demand_coefficients,
                                add_city_pair_seat_flights=seat_flights_per_year,
                                add_city_pair_exp_utility=test_itin_out_exp_utility
                            ) + al.itin_profit(
                                test_itin_in["fare"],
                                test_itin_in,
                                city_pair_in,
                                city_data.loc[destination_id],
                                city_data.loc[origin_id],
                                fleet_data,
                                aircraft_data,
                                FuelCost_USDperGallon,
                                demand_coefficients,
                                add_city_pair_seat_flights=seat_flights_per_year,
                                add_city_pair_exp_utility=test_itin_in_exp_utility
                            )
                        ) / aircraft_data.loc[reassign_ac["SizeClass"], "Seats"]  # seats on this itinerary are only provided by reassigned aircraft

                        # reset fleet_data to avoid mutability issues
                        fleet_data.loc[reassign_ac["AircraftID"], "Flights_perYear"] = old_ac_flights_per_year

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
    op_hrs_per_year: float,
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
    op_hrs_per_year : float
        Number of operating hours per year
    
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

        # adjust city_pair_data Seat_Flights_perYear
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
        else:
            airline_routes[airline_id].loc[out_reassign_mask, "exp_utility"] = 0.0
            airline_routes[airline_id].loc[in_reassign_mask, "exp_utility"] = 0.0
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
                "fuel_stop": -1
            }
            new_itin_in = {
                "origin": new_destination,
                "destination": new_origin,
                "fare": new_city_pair_in["Mean_Fare_USD"],
                "aircraft_ids": [reassign_ac["AircraftID"]],
                "flights_per_year": addnl_flights_per_year,
                "seat_flights_per_year": addnl_seat_flights_per_year,
                "exp_utility": 0.0,  # calculated later
                "fuel_stop": -1
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
    else:
        # add aircraft to grounded_acft
        airlines.loc[airline_id, "Grounded_acft"].append(reassign_ac["AircraftID"])

    if not deploying:
        # adjust city_data movements
        city_data.loc[reassign_itin_out["origin"], "Movts_perHr"] -= 2.0 * float(reassign_ac["Flights_perYear"])/op_hrs_per_year
        city_data.loc[reassign_itin_out["destination"], "Movts_perHr"] -= 2.0 * float(reassign_ac["Flights_perYear"])/op_hrs_per_year
        if reassign_itin_out["fuel_stop"] != -1:
            city_data.loc[reassign_itin_out["fuel_stop"], "Movts_perHr"] -= 4.0 * float(reassign_ac["Flights_perYear"])/op_hrs_per_year
        
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
        city_data.loc[new_origin, "Movts_perHr"] += 2.0 * float(addnl_flights_per_year)/op_hrs_per_year
        city_data.loc[new_destination, "Movts_perHr"] += 2.0 * float(addnl_flights_per_year)/op_hrs_per_year

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
        grounding = True
    else:
        grounding = False

    if not grounding:
        # check if new itinerary already exists in rtn_flt_df
        if rtn_flt_df[
            (rtn_flt_df["Origin"] == new_origin)
            & (rtn_flt_df["Destination"] == new_destination)
            & (rtn_flt_df["Fuel_Stop"] == -1)
        ].empty:
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

    # recalculate profit per seat of altered itineraries
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
            al.itin_profit(
                updated_reassign_itin_out["fare"].iloc[0],
                updated_reassign_itin_out.iloc[0],
                updated_reassign_city_pair_out,
                city_data.loc[reassign_itin_out["origin"]],
                city_data.loc[reassign_itin_out["destination"]],
                airline_fleets[airline_id],
                aircraft_data,
                FuelCost_USDperGallon,
                demand_coefficients,
                # city_pair_data is already updated
            ) + al.itin_profit(
                updated_reassign_itin_in["fare"].iloc[0],
                updated_reassign_itin_in.iloc[0],
                updated_reassign_city_pair_in,
                city_data.loc[reassign_itin_out["destination"]],
                city_data.loc[reassign_itin_out["origin"]],
                airline_fleets[airline_id],
                aircraft_data,
                FuelCost_USDperGallon,
                demand_coefficients,
                # city_pair_data is already updated
            )
        ) / itin_seats

    if not grounding:
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
            itin_seats += aircraft_data.loc[airline_fleets[airline_id].loc[ac_id, "SizeClass"], "Seats"]
        rtn_flt_df.loc[
            (rtn_flt_df["Origin"] == new_origin)
            & (rtn_flt_df["Destination"] == new_destination)
            & (rtn_flt_df["Fuel_Stop"] == -1),
            "Profit_perSeat"
        ] = (
            al.itin_profit(
                updated_new_itin_out["fare"].iloc[0],
                updated_new_itin_out.iloc[0],
                updated_new_city_pair_out,
                city_data.loc[new_origin],
                city_data.loc[new_destination],
                airline_fleets[airline_id],
                aircraft_data,
                FuelCost_USDperGallon,
                demand_coefficients,
                # city_pair_data is already updated
            ) + al.itin_profit(
                updated_new_itin_in["fare"].iloc[0],
                updated_new_itin_in.iloc[0],
                updated_city_pair_in,
                city_data.loc[new_destination],
                city_data.loc[new_origin],
                airline_fleets[airline_id],
                aircraft_data,
                FuelCost_USDperGallon,
                demand_coefficients,
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
    if not grounding:
        rtn_flt_new_itin_mask = (
            (rtn_flt_df["Origin"] == new_origin)
            & (rtn_flt_df["Destination"] == new_destination)
            & (rtn_flt_df["Fuel_Stop"] == -1)
        )
        # aircraft will already exist in new itinerary if it is new to the airline
        if reassign_ac["AircraftID"] not in rtn_flt_df.loc[rtn_flt_new_itin_mask, "Aircraft_IDs"].iloc[0]:
            rtn_flt_df.loc[rtn_flt_new_itin_mask, "Aircraft_IDs"].iloc[0].append(reassign_ac["AircraftID"])
    
    return rtn_flt_df
