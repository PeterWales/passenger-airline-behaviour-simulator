from dataclasses import dataclass


@dataclass
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
        Maximum payload capacity of aircraft in kg
    max_takeoff_kg : float
        Maximum takeoff weight of aircraft in kg
    cruise_ms : float
        Cruise true airspeed of aircraft in m/s
    ceiling_ft : float
        Service ceiling of aircraft in feet
    
    Methods
    -------
    annual_update()
        Update lease cost and age of aircraft
    """

    aircraft_id: int
    seats: int
    lease_USDpermonth: float
    lease_annualmultiplier: float
    age: int
    operation_USDperhr: float
    pilot_USDperhr: float  # per pilot
    crew_USDperhr: float  # per cabin crew
    maintenance_daysperyear: int
    turnaround_hrs: float
    op_empty_kg: float
    max_fuel_kg: float
    max_payload_kg: float  # including passengers and cargo, not fuel or crew
    max_takeoff_kg: float
    cruise_ms: float  # TAS in m/s
    ceiling_ft: float

    def annual_update(self) -> None:
        self.lease_USDpermonth *= self.lease_annualmultiplier
        self.age += 1
