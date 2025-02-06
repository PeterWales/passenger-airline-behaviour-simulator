from dataclasses import dataclass
import numpy as np
import pandas as pd


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
    base_population : float
    population : float
    base_income_USDpercap : float
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
    base_population: float
    population: float
    base_income_USDpercap: float
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
    """Class for keeping track of airline data"""

    airline_id: int
    region: int
    country: str
    country_num: int
    n_aircraft: list


class Route:
    """
    Class for keeping track of route data

    Attributes
    ----------
    route_id : int
    distance : float
        Great circle distance in meters between origin and destination cities
    waypoints : list
        List of dictionaries with latitude and longitude keys
    origin : City
        Instance of City dataclass for the origin city
    destination : City
        Instance of City dataclass for the destination city
    base_demand : int
        Passengers per year
    base_fare : float
        Fare in USD per passenger
    price_elasticity : dict
        Dictionary with keys "route" and "national"

    Methods
    -------
    calc_route(origin: City, destination: City, elasticities: pd.Series) -> tuple[float, list, dict]
        Calculate the great circle route between the origin and destination cities
        Returns the distance in meters, a list of waypoints and a dictionary of price elasticities
    """

    def __init__(
        self,
        route_id: int,
        origin: City,
        destination: City,
        base_demand: int,
        base_fare: float,
        price_elasticities: pd.Series,
        population_elasticity: float,
    ):
        self.route_id = route_id
        self.origin = origin
        self.destination = destination
        self.base_demand = base_demand
        self.mean_demand = base_demand
        self.base_fare = base_fare
        self.mean_fare = base_fare
        self.price_elasticities = price_elasticities
        self.population_elasticity = population_elasticity
        self.static_demand_factor = 1.0  # due to no change in GDP or population until after first year
        self.distance = None
        self.waypoints = None
        self.origin_income_elasticity = None
        self.destination_income_elasticity = None

    def update_route(self) -> None:
        """
        Use Haversine formula to calculate the great circle distance and route between the origin and destination cities.
        """
        # TODO: generate waypoints between origin and destination to route around airspace restrictions

        self.waypoints = [
            {"latitude": self.origin.latitude, "longitude": self.origin.longitude},
            {"latitude": self.destination.latitude, "longitude": self.destination.longitude},
        ]

        r = 6378000.0  # mean radius of the earth in meters
        self.distance = 0.0
        for wpt_num in range(len(self.waypoints) - 1):
            term1 = 1.0 - np.cos(
                np.deg2rad(
                    self.waypoints[wpt_num+1]["latitude"]
                    - self.waypoints[wpt_num]["latitude"]
                )
            )
            term2 = (
                np.cos(np.deg2rad(self.waypoints[wpt_num]["latitude"]))
                * np.cos(np.deg2rad(self.waypoints[wpt_num+1]["latitude"]))
                * (1 - np.cos(np.deg2rad(self.waypoints[wpt_num+1]["longitude"] - self.waypoints[wpt_num]["longitude"])))
            )
            # Haversine Formula can result in numerical errors when origin and destination approach opposite sides of the earth
            self.distance += min(2.0 * r * np.asin(np.sqrt((term1 + term2) / 2)), np.pi * r)

    def update_price_elasticity(self) -> None:
        """
        Determine whether the route is long or short haul and assign the appropriate elasticity values.
        """
        if self.distance < 3000 * 1609.344:  # arbitrary threshold for short/long haul (3000 miles)
            haul = "SH"
        else:
            haul = "LH"
        self.price_elasticities["route"] = self.price_elasticities[f"route_{haul}"]
        self.price_elasticities["national"] = self.price_elasticities[f"national_{haul}"]

    def update_income_elasticity(self, income_elasticities: pd.DataFrame) -> None:
        """
        Determine whether the route is short/medium/long/ultra long haul.
        Determine whether the origin and destination are in the U.S., another developed country or a developing country.
        Assign the appropriate income elasticity values.
        """
        # TODO: determine U.S. country code programmatically
        # TODO: add a static function to avoid code duplication

        if self.distance < 1000 * 1609.344:  # arbitrary threshold for short/medium haul (1000 miles)
            haul = "SH"
        elif self.distance < 3000 * 1609.344:  # arbitrary threshold for medium/long haul (3000 miles)
            haul = "MH"
        elif self.distance < 5000 * 1609.344:  # arbitrary threshold for long/ultra long haul (5000 miles)
            haul = "LH"
        else:
            haul = "ULH"

        # World Bank GNI thresholds for country income levels
        lower_middle_income = 1166
        upper_middle_income = 4526
        high_income = 14005

        # origin
        if self.origin.country == 304:  # U.S. country code
            self.origin_income_elasticity = income_elasticities.loc[
                income_elasticities["CountryType"] == "US", f"elasticity_{haul}"
            ]
        elif self.origin.income_USDpercap < lower_middle_income:
            self.origin_income_elasticity = income_elasticities.loc[
                income_elasticities["CountryType"] == "Developing", f"elasticity_{haul}"
            ]
        elif self.origin.income_USDpercap < high_income:
            # linearly interpolate between developing and developed country elasticities
            gradient = (
                income_elasticities.loc[
                    income_elasticities["CountryType"] == "Developed",
                    f"elasticity_{haul}",
                ]
                - income_elasticities.loc[
                    income_elasticities["CountryType"] == "Developing",
                    f"elasticity_{haul}",
                ]
            ) / (high_income - lower_middle_income)
            self.origin_income_elasticity = income_elasticities.loc[
                income_elasticities["CountryType"] == "Developing", f"elasticity_{haul}"
            ] + gradient * (self.origin.income_USDpercap - lower_middle_income)
        else:
            self.origin_income_elasticity = income_elasticities.loc[
                income_elasticities["CountryType"] == "Developed", f"elasticity_{haul}"
            ]

        # destination
        if self.destination.country == 304:  # U.S. country code
            self.destination_income_elasticity = income_elasticities.loc[
                income_elasticities["CountryType"] == "US", f"elasticity_{haul}"
            ]
        elif self.destination.income_USDpercap < lower_middle_income:
            self.destination_income_elasticity = income_elasticities.loc[
                income_elasticities["CountryType"] == "Developing", f"elasticity_{haul}"
            ]
        elif self.destination.income_USDpercap < high_income:
            # linearly interpolate between developing and developed country elasticities
            gradient = (
                income_elasticities.loc[
                    income_elasticities["CountryType"] == "Developed",
                    f"elasticity_{haul}",
                ]
                - income_elasticities.loc[
                    income_elasticities["CountryType"] == "Developing",
                    f"elasticity_{haul}",
                ]
            ) / (high_income - lower_middle_income)
            self.destination_income_elasticity = income_elasticities.loc[
                income_elasticities["CountryType"] == "Developing", f"elasticity_{haul}"
            ] + gradient * (self.destination.income_USDpercap - lower_middle_income)
        else:
            self.destination_income_elasticity = income_elasticities.loc[
                income_elasticities["CountryType"] == "Developed", f"elasticity_{haul}"
            ]

    def update_static_demand_factor(self) -> None:
        """
        Calculate the total route demand factor from effects independent of fare.
        """
        # assume demand originating from each city is proportional to the city's population * income
        origin_demand_weight = (
            (self.origin.population * self.origin.income_USDpercap)
            / (self.origin.population * self.origin.income_USDpercap
               + self.destination.population * self.destination.income_USDpercap)
        )
        destination_demand_weight = 1 - origin_demand_weight
        
        income_factor_origin = 1 + (
            ((self.origin.income_USDpercap - self.origin.base_income_USDpercap) 
             / self.origin.base_income_USDpercap) * self.origin_income_elasticity)

        income_factor_destination = 1 + (
            ((self.destination.income_USDpercap - self.destination.base_income_USDpercap) 
             / self.destination.base_income_USDpercap) * self.destination_income_elasticity)

        # total income factor weighted by population * income
        income_factor = (
            origin_demand_weight * income_factor_origin
            + destination_demand_weight * income_factor_destination
        )

        population_factor_origin = 1 + (
            ((self.origin.population - self.origin.base_population) 
             / self.origin.base_population) * self.population_elasticity)

        population_factor_destination = 1 + (
            ((self.destination.population - self.destination.base_population) 
             / self.destination.base_population) * self.population_elasticity)

        # total population factor weighted by population * income
        population_factor = (
            origin_demand_weight * population_factor_origin
            + destination_demand_weight * population_factor_destination
        )

        self.static_demand_factor = income_factor * population_factor

    def update_demand(self) -> None:
        """
        Update the route demand based on fare and annual static factors.
        """
        # TODO: add input for national taxes
        fare_factor = 1 + (((self.mean_fare - self.base_fare) / self.base_fare) * self.price_elasticities["route"])
        # tax_factor = 1 + ((delta_tax / self.mean_fare) * self.price_elasticities["national"])

        self.mean_demand = self.base_demand * fare_factor * self.static_demand_factor
