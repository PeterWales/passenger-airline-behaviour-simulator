import math
import pandas as pd


# def annual_update(self) -> None:
#     """
#     Update lease cost and age of aircraft

#     Updates self.lease_USDpermonth and self.age
#     """
#     self.lease_USDpermonth *= self.lease_annualmultiplier
#     self.age += 1


def calc_ranges(aircraft_data: pd.DataFrame, calendar_year: int) -> list:
    """
    Calculate an approximate max range of each aircraft type in m including reserve fuel with 80% of max payload
    """
    # TODO: improve loiter fuel usage calculation

    payload_proportion = 0.8
    loiter_time_s = 30 * 60  # 30 minutes in seconds
    diversion_distance_m = 100 * 1852  # 100 nautical miles in meters

    ranges = []
    for _, aircraft in aircraft_data.iterrows():
        # calculate landing weight
        landing_weight_kg = (
            aircraft["OEM_kg"] + payload_proportion * aircraft["MaxPayload_kg"]
        )

        # calculate weight before loiter using Breguet range equation
        loiter_distance_m = loiter_time_s * aircraft["CruiseV_ms"]
        breguet_factor = (
            (aircraft["Breguet_gradient"] * calendar_year)
            + aircraft["Breguet_intercept"]
        )

        loiter_weight_kg = landing_weight_kg * math.exp(
            loiter_distance_m / breguet_factor
        )

        # calculate weight before 100NM diversion using Breguet range equation
        diversion_weight_kg = loiter_weight_kg * math.exp(
            diversion_distance_m / breguet_factor
        )

        # calculate take-off weight (max fuel + 80% payload doesn't necessarily reach MTOM)
        takeoff_weight_kg = min(
            [
                aircraft["MTOM_kg"],
                aircraft["OEM_kg"]
                + aircraft["MaxFuel_kg"]
                + (payload_proportion * aircraft["MaxPayload_kg"])
            ]
        )

        # calculate max distance up to diversion point using Breguet range equation
        ranges.append(
            breguet_factor * math.log(takeoff_weight_kg / diversion_weight_kg)
        )
    return ranges


def calc_flights_per_year(
    origin: pd.Series,
    origin_id: int,
    destination: pd.Series,
    destination_id: int,
    aircraft: pd.Series,
    city_pair_data: pd.DataFrame,
    fuel_stop_series: None | pd.Series,
    fuel_stop_id: None | int
) -> int:
    """
    Calculate the number of return flights per year the aircraft can fly on its specified route

    Parameters
    ----------
    origin : pd.Series
        Series of data for origin city
    origin_id : int
    destination : pd.Series
        Series of data for destination city
    destination_id : int
    aircraft : pd.Series
        Series of data for aircraft
    city_pair_data : pd.DataFrame
        DataFrame of data for each city pair
    fuel_stop_series : None | pd.Series
        Series of data for city where aircraft must stop to refuel, or None if non-stop
    fuel_stop_id : int
        ID of city where aircraft must stop to refuel

    Returns
    -------
    flights_per_year : int
    """
    curfew_time = 7  # assume airports are closed between 11pm and 6am

    if fuel_stop_series is None:
        outbound_route = city_pair_data.loc[
            (city_pair_data["OriginCityID"] == origin_id)
            & (city_pair_data["DestinationCityID"] == destination_id)
        ].iloc[0]

        flight_time_hrs = outbound_route["Great_Circle_Distance_m"] / (aircraft["CruiseV_ms"] * 3600)
        mean_one_way_hrs = (
            flight_time_hrs + aircraft["Turnaround_hrs"]
            + (
                sum(
                    [
                        origin["Taxi_Out_mins"],
                        destination["Taxi_In_mins"],
                        destination["Taxi_Out_mins"],
                        origin["Taxi_In_mins"],
                    ]
                ) / (60 * 2)
            )
        )

        if flight_time_hrs > curfew_time:
            # assume aircraft can be airborne through the night to avoid curfew
            legs_per_48hrs = math.floor(48 / mean_one_way_hrs)
        else:
            # assume aircraft must be grounded during curfew
            legs_per_48hrs = math.floor(2*(24 - curfew_time) / mean_one_way_hrs)
    else:
        outbound_leg1 = city_pair_data.loc[
            (city_pair_data["OriginCityID"] == origin_id)
            & (city_pair_data["DestinationCityID"] == fuel_stop_id)
        ].iloc[0]
        outbound_leg2 = city_pair_data.loc[
            (city_pair_data["OriginCityID"] == fuel_stop_id)
            & (city_pair_data["DestinationCityID"] == destination_id)
        ].iloc[0]

        flight_time_1_hrs = (
            outbound_leg1["Great_Circle_Distance_m"]
        ) / (aircraft["CruiseV_ms"] * 3600)
        flight_time_2_hrs = (
            outbound_leg2["Great_Circle_Distance_m"]
        ) / (aircraft["CruiseV_ms"] * 3600)

        mean_one_way_hrs = (
            flight_time_1_hrs + flight_time_2_hrs + 2*aircraft["Turnaround_hrs"]
            + (
                sum(
                    [
                        origin["Taxi_Out_mins"],
                        destination["Taxi_In_mins"],
                        destination["Taxi_Out_mins"],
                        origin["Taxi_In_mins"],
                        2*fuel_stop_series["Taxi_In_mins"],
                        2*fuel_stop_series["Taxi_Out_mins"],
                    ]
                ) / (60 * 2)
            )
        )

        if flight_time_1_hrs > curfew_time or flight_time_2_hrs > curfew_time:
            # assume aircraft can be airborne through the night to avoid curfew
            legs_per_48hrs = math.floor(48 / mean_one_way_hrs)
        else:
            # assume aircraft must be grounded during curfew
            legs_per_48hrs = math.floor(2*(24 - curfew_time) / mean_one_way_hrs)

    days_operational = 365 - aircraft["MaintenanceDays_PerYear"]
    return math.floor((legs_per_48hrs * (days_operational / 2))/2)


def calc_fuel_cost(
    aircraft_type: pd.Series,
    aircraft: pd.Series,
    city_pair: pd.Series,
    pax: int,
    FuelCost_USDperGallon: float
) -> float:
    """
    Calculate fuel consumption for a given route and aircraft type
    Loiter and diversion fuel is assumed to be carried but not used

    Parameters
    ----------
    aircraft_type : pd.Series
    aircraft : pd.Series
    city_pair : pd.Series
    pax : int
    FuelCost_USDperGallon : float

    Returns
    -------
    fuel_cost_USD : float (-1 if fuel consumption is more than max fuel capacity or take-off weight is more than MTOM)
    """
    # TODO: improve loiter fuel usage calculation

    loiter_time_s = 30 * 60  # 30 minutes in seconds
    diversion_distance_m = 100 * 1852  # 100 nautical miles in meters
    passenger_mass_kg = 100  # average passenger mass in kg (including baggage)

    # calculate no-fuel weight
    nofuel_weight_kg = (
        aircraft_type["OEM_kg"] + pax * passenger_mass_kg
    )
    # calculate diversion landing weight (before possible loiter) using Breguet range equation
    loiter_distance_m = loiter_time_s * aircraft_type["CruiseV_ms"]
    loiter_weight_kg = nofuel_weight_kg * math.exp(
        loiter_distance_m / aircraft["BreguetFactor"]
    )
    # calculate planned landing weight (before possible diversion) using Breguet range equation
    landing_weight_kg = loiter_weight_kg * math.exp(
        diversion_distance_m / aircraft["BreguetFactor"]
    )
    # calculate take-off weight
    takeoff_weight_kg = landing_weight_kg * math.exp(
        city_pair["Great_Circle_Distance_m"] / aircraft["BreguetFactor"]
    )

    # calculate fuel consumption
    fuel_consumption_kg = takeoff_weight_kg - landing_weight_kg

    if (
        fuel_consumption_kg > aircraft_type["MaxFuel_kg"]
        or takeoff_weight_kg > aircraft_type["MTOM_kg"]
    ):
        fuel_cost_USDperflt = -1  # flag as infeasible
    else:
        # 1L = 0.264 US Gallons, fuel price in USD/gallon
        fuel_cost_USDperflt = fuel_consumption_kg * 0.264 * FuelCost_USDperGallon
    return fuel_cost_USDperflt


def calc_landing_fee(
    city_pair: pd.Series,
    destination: pd.Series,
    aircraft: pd.Series,
    pax: int,
) -> float:
    """
    Calculate airport fee per landing with a given flight type, aircraft type and number of passengers
    """
    if city_pair["international"]:
        flight_type = "International"
    else:
        flight_type = "Domestic"

    mvmt_fee = destination[f"{flight_type}_Fees_USDperMovt"][aircraft["SizeClass"]]
    pax_fee = destination[f"{flight_type}_Fees_USDperPax"] * pax
    fee_perflt = mvmt_fee + pax_fee
    return fee_perflt


def calc_op_cost(
    aircraft: pd.Series,
    city_pair: pd.Series,
    origin: pd.Series,
    destination: pd.Series,
) -> float:
    """
    Calculate operational and crew cost per flight
    """
    # TODO: define magic numbers as constants
    flight_time_hrs = (city_pair["Great_Circle_Distance_m"] / aircraft["CruiseV_ms"]) / 3600.0
    ground_time_hrs = origin["Taxi_Out_mins"] + destination["Taxi_In_mins"] + aircraft["Turnaround_hrs"]

    if flight_time_hrs < 8:
        n_pilots = 2
    elif flight_time_hrs < 12:
        n_pilots = 3
    else:
        n_pilots = 4

    if flight_time_hrs < 10:
        n_cc = math.ceil(aircraft["Seats"] / 50)
    else:
        n_cc = math.ceil((aircraft["Seats"] * 1.5) / 50)

    op_cost_perflt = (
        n_pilots * aircraft["PilotCost_USDperhr"]
        + n_cc * aircraft["CrewCost_USDPerCrewPerHour"]
        + aircraft["OpCost_USDPerHr"]
    ) * (flight_time_hrs + ground_time_hrs) + (
        aircraft["OpCost_USDPerHr"] * flight_time_hrs
    )

    return op_cost_perflt


def calc_lease_cost(aircraft: pd.Series) -> float:
    """
    Calculate the lease cost per flight
    """
    # note aircraft["FlightsPerYear"] is number of return journeys => equal to number of outbound flights
    lease_cost_perflt = aircraft["Lease_USDpermonth"] * 12 / aircraft["FlightsPerYear"]
    return lease_cost_perflt


def calc_flight_cost(
    aircraft_type: pd.Series,
    aircraft: pd.Series,
    city_pair: pd.Series,
    origin: pd.Series,
    destination: pd.Series,
    pax: int,
    FuelCost_USDperGallon: float,
) -> float:
    cost = (
        calc_op_cost(aircraft_type, city_pair, origin, destination)
        + calc_landing_fee(city_pair, destination, aircraft_type, pax)
        + calc_fuel_cost(aircraft_type, aircraft, city_pair, pax, FuelCost_USDperGallon)
        + calc_lease_cost(aircraft)
    )
    return cost
