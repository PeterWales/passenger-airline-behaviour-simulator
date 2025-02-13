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
        # takeoff_weight_kg = min(self.max_takeoff_kg, self.op_empty_kg + self.max_fuel_kg + 0.8*self.max_payload_kg)
        takeoff_weight_kg = min(
            aircraft["MTOM_kg"],
            aircraft["OEM_kg"]
            + aircraft["MaxFuel_kg"]
            + (payload_proportion * aircraft["MaxPayload_kg"]),
        )

        # calculate max distance up to diversion point using Breguet range equation
        ranges.append(
            breguet_factor * math.log(takeoff_weight_kg / diversion_weight_kg)
        )
    return ranges

def calc_flights_per_year(
    origin: pd.Series,
    destination: pd.Series,
    aircraft: pd.Series,
    city_pair_data: pd.DataFrame,
    fuel_stop: None | pd.Series,
) -> int:
    """
    Calculate the number of return flights per year the aircraft can fly on its specified route

    Parameters
    ----------
    fuel_stop : None | pd.Series
        Series of data for city where aircraft must stop to refuel, or None if non-stop
    origin : pd.Series
        Series of data for origin city
    destination : pd.Series
        Series of data for destination city
    aircraft : pd.Series
        Series of data for aircraft
    city_pair_data : pd.DataFrame
        DataFrame of data for each city pair

    Returns
    -------
    flights_per_year : int
    """
    curfew_time = 7  # assume airports are closed between 11pm and 6am

    if fuel_stop is None:
        outbound_route = city_pair_data.loc[
            (city_pair_data["Origin"] == origin["CityID"])
            & (city_pair_data["Destination"] == destination["CityID"])
        ].iloc[0]

        flight_time_hrs = outbound_route["Great_Circle_Distance_m"] / (aircraft["CruiseV_ms"] * 3600)
        mean_one_way_hrs = (
            flight_time_hrs + aircraft["Turnaround_hrs"]
            + (
                sum(
                    origin["Taxi_Out_mins"],
                    destination["Taxi_In_mins"],
                    destination["Taxi_Out_mins"],
                    origin["Taxi_In_mins"],
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
            (city_pair_data["Origin"] == origin["CityID"])
            & (city_pair_data["Destination"] == fuel_stop["CityID"])
        ].iloc[0]
        outbound_leg2 = city_pair_data.loc[
            (city_pair_data["Origin"] == fuel_stop["CityID"])
            & (city_pair_data["Destination"] == destination["CityID"])
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
                    origin["Taxi_Out_mins"],
                    destination["Taxi_In_mins"],
                    destination["Taxi_Out_mins"],
                    origin["Taxi_In_mins"],
                    2*fuel_stop["Taxi_In_mins"],
                    2*fuel_stop["Taxi_Out_mins"],
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
