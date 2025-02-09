from city import City
import numpy as np
import pandas as pd
from tqdm import tqdm


class Route:
    """
    Class for keeping track of route data

    Attributes
    ----------
    route_id : int
    origin : City
        Instance of City dataclass for origin city
    destination : City
        Instance of City dataclass for destination city
    base_demand : int
        Base year demand for the route in passengers
    current_demand : int
        Current year demand for the route in passengers
        Total demand for all itineraries
    base_mean_fare : float
        Base year mean fare (all itineraries) for the route in USD
    mean_fare : float
        Current year mean fare (all itineraries) for the route in USD
    price_elasticities : pd.Series
        Price elasticity of demand values for the route
        Initially contains fields:
            route_SH, route_LH, national_SH, national_LH
        Additional fields are added upon first call to update_price_elasticity:
            route, national
    population_elasticity : float
        Population elasticity of demand for the route
    static_demand_factor : float
        Factor for demand independent of fare, valid for the current year
    distance : float
        Great circle distance between origin and destination cities in meters
    waypoints : list
        List of dictionaries containing latitude and longitude of waypoints
    origin_income_elasticity : float
        Income elasticity of demand for the origin city
    destination_income_elasticity : float
        Income elasticity of demand for the destination city

    Methods
    -------
    initialise_routes(
        cities: list,
        city_pair_data: pd.DataFrame,
        price_elasticities: pd.DataFrame,
        income_elasticities: pd.DataFrame,
        population_elasticity: float
    ) -> list
        Generate 2D list of instances of Route dataclass from cities and contents of DataByCityPair and Elasticities files
    
    update_route() -> None
        Use Haversine formula to calculate the great circle distance and route between the origin and destination cities
    
    update_price_elasticity() -> None
        Determine whether the route is long or short haul and assign the appropriate elasticity values
    
    update_income_elasticity(income_elasticities: pd.DataFrame) -> None
        Determine whether the route is short/medium/long/ultra long haul.
        Determine whether the origin and destination are in the U.S., another developed country or a developing country.
        Assign the appropriate income elasticity values

    update_static_demand_factor() -> None
        Calculate the total route demand factor from effects independent of fare
    
    update_demand() -> None
        Update the route demand based on fare and annual static factors
    """

    def __init__(
        self,
        route_id: int,
        origin: City,
        destination: City,
        base_demand: int,
        base_mean_fare: float,
        price_elasticities: pd.Series,
        population_elasticity: float,
    ):
        self.route_id = route_id
        self.origin = origin
        self.destination = destination
        self.base_demand = base_demand
        self.current_demand = base_demand
        self.base_mean_fare = base_mean_fare
        self.mean_fare = base_mean_fare
        self.price_elasticities = price_elasticities
        self.population_elasticity = population_elasticity
        self.static_demand_factor = 1.0  # due to no change in GDP or population until after first year
        self.distance = None
        self.waypoints = None
        self.origin_income_elasticity = None
        self.destination_income_elasticity = None

    @staticmethod
    def initialise_routes(
        cities: list,
        city_pair_data: pd.DataFrame,
        price_elasticities: pd.DataFrame,
        income_elasticities: pd.DataFrame,
        population_elasticity: float,
    ) -> list:
        """
        Generate 2D list of instances of Route dataclass from cities and contents of DataByCityPair and Elasticities files

        Parameters
        ----------
        cities : list of instances of City dataclass
        city_pair_data : pd.DataFrame
        price_elasticities : pd.DataFrame
        income_elasticities : pd.DataFrame
        population_elasticity : float

        Returns
        -------
        routes : 2D list of instances of Route dataclass, indexed by [OriginCityID, DestinationCityID]
        """
        # order price_elasticities dataframe by OD_1 and OD_2
        price_elasticities = price_elasticities.sort_values(by=["OD_1", "OD_2"])

        n_cities = len(cities)
        routes = [[None for _ in range(n_cities)] for _ in range(n_cities)]

        route_id = 0

        # show a progress bar because this step can take a while
        for idx, route in tqdm(
            city_pair_data.iterrows(),
            total=city_pair_data.shape[0],
            desc="        Routes created",
            ascii=False,
            ncols=75,
        ):
            origin_id = int(route["OriginCityID"])
            destination_id = int(route["DestinationCityID"])

            if origin_id == 0 or destination_id == 0 or origin_id == destination_id:
                continue

            # find the elasticity values for the current route - note OD_1 <= OD_2
            price_elasticities_series = price_elasticities.loc[
                (
                    price_elasticities["OD_1"]
                    == min(cities[origin_id].region, cities[destination_id].region)
                )
                & (
                    price_elasticities["OD_2"]
                    == max(cities[origin_id].region, cities[destination_id].region)
                )
            ]
            # convert from DataFrame to Series
            price_elasticities_series = price_elasticities_series.squeeze()

            routes[origin_id][destination_id] = Route(
                route_id=route_id,
                origin=cities[origin_id],
                destination=cities[destination_id],
                base_demand=route["BaseYearODDemandPax_Est"],
                base_mean_fare=route["Fare_Est"],
                price_elasticities=price_elasticities_series,
                population_elasticity=population_elasticity,
            )
            # calculate distance and save waypoints
            routes[origin_id][destination_id].update_route()
            # calculate initial elasticities and static demand factor
            routes[origin_id][destination_id].update_price_elasticity()
            routes[origin_id][destination_id].update_income_elasticity(income_elasticities)
            routes[origin_id][destination_id].update_static_demand_factor()

            route_id += 1
        return routes

    def update_route(self) -> None:
        """
        Use Haversine formula to calculate the great circle distance and route between the origin and destination cities.
        Updates self.distance and self.waypoints.
        """
        # TODO: generate waypoints between origin and destination to route around airspace restrictions

        self.waypoints = [
            {
                "latitude": self.origin.latitude,
                "longitude": self.origin.longitude
            },
            {
                "latitude": self.destination.latitude,
                "longitude": self.destination.longitude,
            },
        ]

        r = 6378000.0  # mean radius of the earth in meters
        self.distance = 0.0
        for wpt_num in range(len(self.waypoints) - 1):
            term1 = 1.0 - np.cos(
                np.deg2rad(
                    self.waypoints[wpt_num + 1]["latitude"]
                    - self.waypoints[wpt_num]["latitude"]
                )
            )
            term2 = (
                np.cos(np.deg2rad(self.waypoints[wpt_num]["latitude"]))
                * np.cos(np.deg2rad(self.waypoints[wpt_num+1]["latitude"]))
                * (1 - np.cos(np.deg2rad(
                    self.waypoints[wpt_num+1]["longitude"]
                    - self.waypoints[wpt_num]["longitude"]))
                )
            )
            # Haversine Formula can result in numerical errors when origin and destination approach opposite sides of the earth
            self.distance += min(
                2.0 * r * np.asin(np.sqrt((term1 + term2) / 2)),
                np.pi * r
            )

    def update_price_elasticity(self) -> None:
        """
        Determine whether the route is long or short haul and assign the appropriate elasticity values.
        Updates/adds self.price_elasticities["route"] and self.price_elasticities["national"].
        """
        if (self.distance < 3000 * 1609.344):
            # arbitrary threshold for short/long haul (3000 miles)
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

        Parameters
        ----------
        income_elasticities : pd.DataFrame

        Updates self.origin_income_elasticity and self.destination_income_elasticity.
        """
        # TODO: determine U.S. country code programmatically
        # TODO: add a static function to avoid code duplication

        if self.distance < 1000 * 1609.344:
            # arbitrary threshold for short/medium haul (1000 miles)
            haul = "SH"
        elif self.distance < 3000 * 1609.344:
            # arbitrary threshold for medium/long haul (3000 miles)
            haul = "MH"
        elif self.distance < 5000 * 1609.344:
            # arbitrary threshold for long/ultra long haul (5000 miles)
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

        Updates self.static_demand_factor.
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

        Updates self.current_demand.
        """
        # TODO: add input for national taxes
        fare_factor = 1 + (
            ((self.mean_fare - self.base_mean_fare) / self.base_mean_fare)
            * self.price_elasticities["route"]
        )
        # tax_factor = 1 + ((delta_tax / self.mean_fare) * self.price_elasticities["national"])

        self.current_demand = self.base_demand * fare_factor * self.static_demand_factor
