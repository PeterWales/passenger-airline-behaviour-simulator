from dataclasses import dataclass


@dataclass
class City:
    """Class for keeping track of city data."""
    city_id: int
    city_name: str
    region: int
    country: int
    local_region: int
    capital_city: int
    population: float
    income_pc: float
    latitude: float
    longitude: float
    domestic_fees_perpax: list
    domestic_fees_permvmt: list
    international_fees_perpax: list
    international_fees_permvmt: list
    taxi_out_mins: float
    taxi_in_mins: float
    capacity_perhr: float
