from dataclasses import dataclass


@dataclass
class City:
    """
    Class for keeping track of city data.

    Attributes
    ----------
    city_id : int
    city_name : str
    region : int
    country : int
    local_region : int
    capital_city : bool
    population : float
    income_USDpercap : float
        Mean income per capita in USD
    latitude : float
    longitude : float
    domestic_fees_USDperpax : float
        landing fees in USD per passenger on an incoming domestic flight
    domestic_fees_USDpermvmt : list
        landing fees in USD per landing for an incoming domestic flight
    international_fees_USDperpax : float
        landing fees in USD per passenger on an incoming international flight
    international_fees_USDpermvmt : list
        landing fees in USD per landing for an incoming international flight
    taxi_out_mins : float
        Average time in minutes to get from gate to takeoff
    taxi_in_mins : float
        Average time in minutes to get from landing to gate
    capacity_perhr : float
        Maximum aircraft movements (takeoffs and landings) in any one hour
    """
    city_id: int
    city_name: str
    region: int
    country: int
    local_region: int
    capital_city: int
    population: float
    income_USDpercap: float
    latitude: float
    longitude: float
    domestic_fees_USDperpax: float
    domestic_fees_USDpermvmt: list
    international_fees_USDperpax: float
    international_fees_USDpermvmt: list
    taxi_out_mins: float
    taxi_in_mins: float
    capacity_perhr: float


@dataclass
class Aircraft:
    """Class for keeping track of aircraft data"""
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


@dataclass
class Airline:
    """ Class for keeping track of airline data"""
    airline_id: int
    region: int
    country: str
    country_num: int
    n_aircraft: list
