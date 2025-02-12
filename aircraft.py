from dataclasses import dataclass
import math
import pandas as pd


class Aircraft:
    """
    Class for keeping track of aircraft data
    
    Attributes
    ----------
    aircraft_id : int
        ID of aircraft
    aircraft_size : int
        Size class of aircraft
    seats : int
        Number of seats in aircraft
    lease_USDpermonth : float
        Cost of leasing aircraft per month in USD
    lease_annualmultiplier : float
        Multiplier for lease cost from year to year
    age : int
        Age of aircraft in years
    operation_USDperhr : float
        Cost of operating aircraft per hour in USD
    pilot_USDperhr : float
        Cost of paying each pilot per hour in USD
    crew_USDperhr : float
        Cost of paying each cabin crew per hour in USD
    maintenance_daysperyear : int
        Number of days per year aircraft is in maintenance
    turnaround_hrs : float
        Time in hours for aircraft to turnaround at gate
    op_empty_kg : float
        Operating empty weight of aircraft in kg
    max_fuel_kg : float
        Maximum fuel capacity of aircraft in kg
    max_payload_kg : float
        Maximum payload capacity of aircraft in kg (not including fuel or crew)
    max_takeoff_kg : float
        Maximum takeoff weight of aircraft in kg
    cruise_ms : float
        Cruise true airspeed of aircraft in m/s
    ceiling_ft : float
        Service ceiling of aircraft in feet
    breguet_factor : float
        (V * L/D) / SFC of aircraft
    breguet_gradient : float
        Gradient of Breguet factor with respect to calendar year
    retirement_age : int
        Age at which aircraft is retired
    range_m : None
        Range of aircraft in m including reserve fuel with 80% of max payload
    route_origin : int
        Origin city ID of route
    route_destination : int
        Destination city ID of route
    flights_per_year : int
        Number of return flights per year the aircraft can fly on its specified route
    fuel_stop : None | int
        City ID of city where aircraft must stop to refuel, or None if non-stop
    
    Methods
    -------
    annual_update()
        Update lease cost and age of aircraft

    calc_ranges()
        Calculate an approximate max range of each aircraft type in m
    """
    #TODO: improve init method to take a pd series of aircraft data as input
    def __init__(
        self,
        aircraft_id: int,
        aircraft_size: int,
        seats: int,
        lease_new_USDpermonth: float,
        lease_annualmultiplier: float,
        age: int,
        operation_USDperhr: float,
        pilot_USDperhr: float,
        crew_USDperhr: float,
        maintenance_daysperyear: int,
        turnaround_hrs: float,
        op_empty_kg: float,
        max_fuel_kg: float,
        max_payload_kg: float,
        max_takeoff_kg: float,
        cruise_ms: float,
        ceiling_ft: float,
        breguet_gradient: float,
        breguet_intercept: float,
        retirement_age: int,
        calendar_year: int,
        route_origin: int,
        route_destination: int,
        fuel_stop: None | int = None,
    ) -> None:
        self.aircraft_id = aircraft_id
        self.aircraft_size = aircraft_size
        self.seats = seats
        self.lease_USDpermonth = lease_new_USDpermonth * (lease_annualmultiplier ** age)
        self.lease_annualmultiplier = lease_annualmultiplier
        self.age = age
        self.operation_USDperhr = operation_USDperhr
        self.pilot_USDperhr = pilot_USDperhr
        self.crew_USDperhr = crew_USDperhr
        self.maintenance_daysperyear = maintenance_daysperyear
        self.turnaround_hrs = turnaround_hrs
        self.op_empty_kg = op_empty_kg
        self.max_fuel_kg = max_fuel_kg
        self.max_payload_kg = max_payload_kg
        self.max_takeoff_kg = max_takeoff_kg
        self.cruise_ms = cruise_ms
        self.ceiling_ft = ceiling_ft
        self.breguet_factor = (breguet_gradient * calendar_year) + breguet_intercept
        self.breguet_gradient = breguet_gradient
        self.retirement_age = retirement_age
        self.route_origin = route_origin
        self.route_destination = route_destination
        self.flights_per_year = None
        self.fuel_stop = fuel_stop

    def annual_update(self) -> None:
        """
        Update lease cost and age of aircraft

        Updates self.lease_USDpermonth and self.age
        """
        self.lease_USDpermonth *= self.lease_annualmultiplier
        self.age += 1

    @staticmethod
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

    def update_flights_per_year(self, routes: list) -> None:
        """
        Calculate the number of return flights per year the aircraft can fly on its specified route

        Parameters
        ----------
        routes : list
            List of routes between cities
        
        Updates self.flights_per_year
        """
        curfew_time = 7  # assume airports are closed between 11pm and 6am

        if self.fuel_stop is None:
            flight_time_hrs = routes[self.route_origin][self.route_destination].distance / (self.cruise_ms * 3600)
            mean_one_way_hrs = (
                flight_time_hrs + self.turnaround_hrs
                + (
                    sum(
                        routes[self.route_origin][self.route_destination].origin.taxi_out_mins,
                        routes[self.route_origin][self.route_destination].destination.taxi_in_mins,
                        routes[self.route_origin][self.route_destination].destination.taxi_out_mins,
                        routes[self.route_origin][self.route_destination].origin.taxi_in_mins,
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
            flight_time_1_hrs = (
                routes[self.route_origin][self.fuel_stop].distance
            ) / (self.cruise_ms * 3600)
            flight_time_2_hrs = (
                routes[self.fuel_stop][self.route_destination].distance
            ) / (self.cruise_ms * 3600)

            mean_one_way_hrs = (
                flight_time_1_hrs + flight_time_2_hrs + 2*self.turnaround_hrs
                + (
                    sum(
                        routes[self.route_origin][self.route_destination].origin.taxi_out_mins,
                        routes[self.route_origin][self.route_destination].destination.taxi_in_mins,
                        routes[self.route_origin][self.route_destination].destination.taxi_out_mins,
                        routes[self.route_origin][self.route_destination].origin.taxi_in_mins,
                        2*routes[self.route_origin][self.fuel_stop].destination.taxi_in_mins,
                        2*routes[self.route_origin][self.fuel_stop].destination.taxi_out_mins,
                    ) / (60 * 2)
                )
            )

            if flight_time_1_hrs > curfew_time or flight_time_2_hrs > curfew_time:
                # assume aircraft can be airborne through the night to avoid curfew
                legs_per_48hrs = math.floor(48 / mean_one_way_hrs)
            else:
                # assume aircraft must be grounded during curfew
                legs_per_48hrs = math.floor(2*(24 - curfew_time) / mean_one_way_hrs)

        days_operational = 365 - self.maintenance_daysperyear
        self.flights_per_year = math.floor((legs_per_48hrs * (days_operational / 2))/2)
