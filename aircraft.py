from dataclasses import dataclass
import math


class Aircraft:
    """
    Class for keeping track of aircraft data
    
    Attributes
    ----------
    aircraft_id : int
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
    
    Methods
    -------
    annual_update()
        Update lease cost and age of aircraft

    calc_range()
        Calculate the range of the aircraft in m including reserve fuel with 80% of max payload
    """

    def __init__(
            self,
            aircraft_id: int,
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
        ) -> None:
            self.aircraft_id = aircraft_id
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
            self.range_m = None

    def annual_update(self) -> None:
        """
        Update lease cost and age of aircraft

        Updates self.lease_USDpermonth, self.breguet_factor and self.age
        """
        self.lease_USDpermonth *= self.lease_annualmultiplier
        self.breguet_factor += self.breguet_gradient
        self.age += 1

    def calc_range(self) -> None:
        """
        Calculate the range of the aircraft in m including reserve fuel with 80% of max payload

        Updates self.range_m
        """
        # TODO: improve loiter fuel usage calculation

        # calculate landing weight
        landing_weight_kg = self.op_empty_kg + 0.8*self.max_payload_kg

        # calculate weight before 30 min loiter using Breguet range equation
        loiter_distance_m = 30*60*self.cruise_ms
        loiter_weight_kg = landing_weight_kg * math.exp(loiter_distance_m / self.breguet_factor)

        # calculate weight before 100NM diversion using Breguet range equation
        diversion_distance_m = 100*1852
        diversion_weight_kg = loiter_weight_kg * math.exp(diversion_distance_m / self.breguet_factor)

        # calculate take-off weight (max fuel + 80% payload doesn't necessarily reach MTOM)
        takeoff_weight_kg = min(self.max_takeoff_kg, self.op_empty_kg + self.max_fuel_kg + 0.8*self.max_payload_kg)

        # calculate max distance up to diversion point using Breguet range equation
        self.range_m = self.breguet_factor * math.log(takeoff_weight_kg / diversion_weight_kg)
