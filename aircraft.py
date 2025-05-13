import math
import pandas as pd
from constants import (
    MAX_RANGE_PAYLOAD_PROPORTION,
    LOITER_T_SEC,
    DIVERSION_DIST_METRES,
    PASSENGER_MASS_KG,
    CURFEW_HOURS,
    FUEL_GALLONS_PER_KG,
)

def calc_ranges(aircraft_data: pd.DataFrame, calendar_year: int) -> list:
    """
    Calculate an approximate max range of each aircraft type in m including reserve fuel with 80% of max payload
    """
    # TODO: improve loiter fuel usage calculation

    ranges = []
    for _, aircraft in aircraft_data.iterrows():
        # calculate landing weight
        landing_weight_kg = (
            aircraft["OEM_kg"] + MAX_RANGE_PAYLOAD_PROPORTION * aircraft["MaxPayload_kg"]
        )

        # calculate weight before loiter using Breguet range equation
        loiter_distance_m = LOITER_T_SEC * aircraft["CruiseV_ms"]
        breguet_factor = (
            (aircraft["Breguet_gradient"] * calendar_year)
            + aircraft["Breguet_intercept"]
        )

        loiter_weight_kg = landing_weight_kg * math.exp(
            loiter_distance_m / breguet_factor
        )

        # calculate weight before 100NM diversion using Breguet range equation
        diversion_weight_kg = loiter_weight_kg * math.exp(
            DIVERSION_DIST_METRES / breguet_factor
        )

        # calculate take-off weight (max fuel + 80% payload doesn't necessarily reach MTOM)
        takeoff_weight_kg = min(
            [
                aircraft["MTOM_kg"],
                aircraft["OEM_kg"]
                + aircraft["MaxFuel_kg"]
                + (MAX_RANGE_PAYLOAD_PROPORTION * aircraft["MaxPayload_kg"])
            ]
        )

        # calculate max distance up to diversion point using Breguet range equation
        ranges.append(
            breguet_factor * math.log(takeoff_weight_kg / diversion_weight_kg)
        )
    return ranges


def calc_itin_time(
    itinerary: pd.Series,
    city_data: pd.DataFrame,
    city_pair_data: pd.DataFrame,
    aircraft_data: pd.DataFrame,
    fleet_data: pd.DataFrame,
) -> float:
    """
    Calculate the time taken to complete a given itinerary (one-way)
    """
    fuel_stop_id = itinerary["fuel_stop"]
    origin_id = itinerary["origin"]
    destination_id = itinerary["destination"]

    # calculate mean cruise speed and turnaround time for aircraft assigned to itinerary
    n_ac = len(itinerary["aircraft_ids"])
    speed_sum = 0
    turnaround_sum = 0
    for ac in itinerary["aircraft_ids"]:
        ac_type = fleet_data.loc[fleet_data["AircraftID"] == ac].iloc[0]["SizeClass"]
        aircraft = aircraft_data.loc[ac_type]
        speed_sum += aircraft["CruiseV_ms"]
        turnaround_sum += aircraft["Turnaround_hrs"]
    mean_speed = speed_sum / n_ac
    mean_turnaround = turnaround_sum / n_ac

    if fuel_stop_id == -1:
        route = city_pair_data.loc[
            (city_pair_data["OriginCityID"] == origin_id)
            & (city_pair_data["DestinationCityID"] == destination_id)
        ].iloc[0]

        flight_time_hrs = route["Great_Circle_Distance_m"] / (mean_speed * 3600)
        itin_time_hrs = (
            flight_time_hrs + mean_turnaround
            + (
                city_data.loc[origin_id, "Taxi_Out_mins"]
                + city_data.loc[destination_id, "Taxi_In_mins"]
            ) / 60.0
        )

    else:
        leg1 = city_pair_data.loc[
            (city_pair_data["OriginCityID"] == origin_id)
            & (city_pair_data["DestinationCityID"] == fuel_stop_id)
        ].iloc[0]
        leg2 = city_pair_data.loc[
            (city_pair_data["OriginCityID"] == fuel_stop_id)
            & (city_pair_data["DestinationCityID"] == destination_id)
        ].iloc[0]

        flight_time_1_hrs = (
            leg1["Great_Circle_Distance_m"]
        ) / (mean_speed * 3600)
        flight_time_2_hrs = (
            leg2["Great_Circle_Distance_m"]
        ) / (mean_speed * 3600)

        itin_time_hrs = (
            flight_time_1_hrs + flight_time_2_hrs + 2*mean_turnaround
            + (
                city_data.loc[origin_id, "Taxi_Out_mins"]
                + city_data.loc[destination_id, "Taxi_In_mins"]
                + city_data.loc[fuel_stop_id, "Taxi_In_mins"]
                + city_data.loc[fuel_stop_id, "Taxi_Out_mins"]
            ) / 60.0
        )

    return itin_time_hrs


def calc_flights_per_year(
    origin: pd.Series,
    origin_id: int,
    destination: pd.Series,
    destination_id: int,
    aircraft: pd.Series,
    city_pair_data: pd.DataFrame,
    fuel_stop_series: None | pd.Series,
    fuel_stop_id: int
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
        ID of city where aircraft must stop to refuel, or -1 if non-stop

    Returns
    -------
    flights_per_year : int
    """
    if fuel_stop_id == -1:
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

        if flight_time_hrs > CURFEW_HOURS:
            # assume aircraft can be airborne through the night to avoid curfew
            legs_per_48hrs = math.floor(48 / mean_one_way_hrs)
        else:
            # assume aircraft must be grounded during curfew
            legs_per_48hrs = math.floor(2*(24 - CURFEW_HOURS) / mean_one_way_hrs)
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

        if flight_time_1_hrs > CURFEW_HOURS or flight_time_2_hrs > CURFEW_HOURS:
            # assume aircraft can be airborne through the night to avoid curfew
            legs_per_48hrs = math.floor(48 / mean_one_way_hrs)
        else:
            # assume aircraft must be grounded during curfew
            legs_per_48hrs = math.floor(2*(24 - CURFEW_HOURS) / mean_one_way_hrs)

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

    # calculate no-fuel weight
    nofuel_weight_kg = (
        aircraft_type["OEM_kg"] + pax * PASSENGER_MASS_KG
    )
    # calculate diversion landing weight (before possible loiter) using Breguet range equation
    loiter_distance_m = LOITER_T_SEC * aircraft_type["CruiseV_ms"]
    loiter_weight_kg = nofuel_weight_kg * math.exp(
        loiter_distance_m / aircraft["BreguetFactor"]
    )
    # calculate planned landing weight (before possible diversion) using Breguet range equation
    landing_weight_kg = loiter_weight_kg * math.exp(
        DIVERSION_DIST_METRES / aircraft["BreguetFactor"]
    )
    # calculate take-off weight
    takeoff_weight_kg = landing_weight_kg * math.exp(
        city_pair["Great_Circle_Distance_m"] / aircraft["BreguetFactor"]
    )

    # calculate fuel consumption
    fuel_consumption_kg = takeoff_weight_kg - landing_weight_kg

    aircraft_type_id = aircraft_type.name
    if fuel_consumption_kg > aircraft_type["MaxFuel_kg"]:
        print("WARNING [calc_fuel_cost]: fuel consumption exceeds max fuel capacity")
        print(f"Itinerary from {city_pair['OriginCityID']} to {city_pair['DestinationCityID']} with aircraft type {aircraft_type_id}")
    if takeoff_weight_kg > aircraft_type["MTOM_kg"]:
        print("WARNING [calc_fuel_cost]: required take-off mass exceeds MTOM")
        print(f"Itinerary from {city_pair['OriginCityID']} to {city_pair['DestinationCityID']} with aircraft type {aircraft_type_id}")
    
    fuel_cost_USDperflt = fuel_consumption_kg * FUEL_GALLONS_PER_KG * FuelCost_USDperGallon
    
    return fuel_cost_USDperflt


def calc_landing_fee(
    city_pair: pd.Series,
    destination: pd.Series,
    aircraft_size: int,
    pax: int,
) -> float:
    """
    Calculate airport fee per landing with a given flight type, aircraft type and number of passengers
    """
    if city_pair["International"]:
        flight_type = "International"
    else:
        flight_type = "Domestic"

    mvmt_fee = destination[f"{flight_type}_Fees_USDperMovt"][aircraft_size]
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
    ground_time_hrs = (origin["Taxi_Out_mins"]/60.0) + (destination["Taxi_In_mins"]/60.0) + aircraft["Turnaround_hrs"]

    # pilots
    if flight_time_hrs < 8:
        n_pilots = 2
    elif flight_time_hrs < 12:
        n_pilots = 3
    else:
        n_pilots = 4

    # cabin crew
    if flight_time_hrs < 10:
        n_cc = math.ceil(aircraft["Seats"] / 50)
    else:
        n_cc = math.ceil((aircraft["Seats"] * 1.5) / 50)

    op_cost_perflt = (
        n_pilots * aircraft["PilotCost_USDPerPilotPerHour"]
        + n_cc * aircraft["CrewCost_USDPerCrewPerHour"]
    ) * (flight_time_hrs + ground_time_hrs) + (
        aircraft["OpCost_USDPerHr"] * flight_time_hrs
    )

    return op_cost_perflt


def calc_lease_cost(aircraft: pd.Series) -> float:
    """
    Calculate the lease cost per flight
    """
    # note aircraft["FlightsPerYear"] is number of return journeys => equal to number of outbound flights
    lease_cost_perflt = float(aircraft["Lease_USDperMonth"]) * 12.0 / aircraft["Flights_perYear"]
    return lease_cost_perflt


def calc_flight_cost(
    # TODO: split this into fixed and variable costs
    airline_route: pd.Series,
    fleet_df: pd.DataFrame,
    aircraft_data: pd.DataFrame,
    city_pair: pd.Series,
    origin: pd.Series,
    destination: pd.Series,
    annual_itin_demand: int,
    FuelCost_USDperGallon: float,
) -> float:
    planes = airline_route["aircraft_ids"]

    # calculate total seats for all aircraft on route
    total_seats = 0
    for acft_id in planes:
        aircraft = fleet_df.loc[fleet_df["AircraftID"] == acft_id].iloc[0]
        aircraft_type = aircraft_data.loc[aircraft["SizeClass"]]
        total_seats += aircraft_type["Seats"]

    # split demand between aircraft based on seat capacity and calculate total cost
    annual_cost = 0
    for acft_id in planes:
        aircraft = fleet_df.loc[fleet_df["AircraftID"] == acft_id].iloc[0]
        aircraft_type = aircraft_data.loc[aircraft["SizeClass"]]

        itin_demand_share = float(aircraft_type["Seats"] * annual_itin_demand) / total_seats
        pax_perflt_share = math.floor(itin_demand_share / aircraft["Flights_perYear"])
        pax = min([pax_perflt_share, aircraft_type["Seats"]])
        pax = max([pax, 0])

        cost_perflt = (
            calc_op_cost(aircraft_type, city_pair, origin, destination)
            + calc_landing_fee(city_pair, destination, int(aircraft["SizeClass"]), pax)
            + calc_fuel_cost(aircraft_type, aircraft, city_pair, pax, FuelCost_USDperGallon)
            + calc_lease_cost(aircraft)
        )
        annual_cost += cost_perflt * aircraft["Flights_perYear"]
    return annual_cost


def annual_update(
    airlines: pd.DataFrame,
    airline_fleets: list[pd.DataFrame],
    aircraft_data: pd.DataFrame,
    year_entering: int,
) -> list[pd.DataFrame]:
    # TODO: when new ac added, check whether efficiency improvement negates the need for a fuel stop and update route appropriately
    for airline_id, airline in airlines.iterrows():
        for _, ac in airline_fleets[airline_id].iterrows():
            ac_type = aircraft_data.loc[ac["SizeClass"]]
            if ac["Age_years"] == ac_type["RetirementAge_years"]:
                # retire aircraft and replace with a new one of the same type
                ac["Age_years"] = 0
                ac["Lease_USDperMonth"] = ac_type["LeaseRateNew_USDPerMonth"]
                ac["BreguetFactor"] = (ac_type["Breguet_gradient"] * year_entering) + ac_type["Breguet_intercept"]
            else:
                ac["Age_years"] += 1
                ac["Lease_USDperMonth"] *= ac_type["LeaseRateAnnualMultiplier"]
    return airline_fleets
