from dataclasses import dataclass
import pandas as pd
from operator import add


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
        Base year population of city
    population : float
        Current year population of city
    base_income_USDpercap : float
        Base year income per capita of city in USD
    income_USDpercap : float
        Current year income per capita of city in USD
    latitude : float
        Capacity-weighted mean latitude of airports in city
    longitude : float
        Capacity-weighted mean longitude of airports in city
    domestic_fees_USDperpax : float
        Capacity-weighted landing fees in USD per passenger on an incoming domestic flight
    domestic_fees_USDpermvmt : list
        Capacity-weighted landing fees in USD per landing for an incoming domestic flight
    international_fees_USDperpax : float
        Capacity-weighted landing fees in USD per passenger on an incoming international flight
    international_fees_USDpermvmt : list
        Capacity-weighted landing fees in USD per landing for an incoming international flight
    taxi_out_mins : float
        Capacity-weighted time in minutes to get from gate to takeoff
    taxi_in_mins : float
        Capacity-weighted time in minutes to get from landing to gate
    capacity_perhr : float
        Sum of maximum aircraft movements (takeoffs and landings) in any one hour for all airports in city
    longest_runway_m : float
        Length of longest runway in city in meters

    Methods
    -------
    initialise_cities(city_data, airport_data)
        Generate list of instances of City dataclass from contents of CityData and AirportData files
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
    longest_runway_m: float

    @staticmethod
    def initialise_cities(city_data: pd.DataFrame, airport_data: pd.DataFrame) -> tuple[list, list]:
        """
        Generate list of instances of City dataclass from contents of CityData and AirportData files
        and a list of lists for looking up which cities are located in which countries.

        cities list is indexed to match CityID field in CityData file
        If a certain index doesn't exist as a CityID in CityData, the corresponding list element is None
        city_lookup list is indexed to match CountryID field in CountryData file, and contains lists of CityIDs
        If a certain country doesn't contain any cities, the corresponding list element is empty

        Parameters
        ----------
        city_data : pd.DataFrame
            sorted by CityID
        airport_data : pd.DataFrame

        Returns
        -------
        cities : list of instances of City dataclass
        city_lookup : list of lists
        """
        # check number of different aircraft supported by airport_data
        n_aircraft = 0
        while (
            f"LandingCosts_PerMovt_Size{n_aircraft}_Domestic_2015USdollars"
            in airport_data.columns
        ):
            n_aircraft += 1

        # initialise cities list
        last_city_id = city_data["CityID"].iloc[-1]
        cities = [None] * (last_city_id + 1)

        # initialise city_lookup list
        last_country_id = max(city_data["Country"])
        city_lookup = [[] for _ in range(last_country_id + 1)]

        # loop over rows of city_data
        for city_id in range(last_city_id + 1):
            # check whether city_id is a CityID in city_data
            if not (city_id in city_data["CityID"].values):
                continue  # leave cities[city_id] as None

            city = city_data.loc[city_data["CityID"] == city_id]
            # convert from DataFrame to Series
            city = city.squeeze()

            # calculate capacity-weighted mean of airport data
            capacity_sum = 0.0
            longest_runway_m = 0.0
            lat_sum = 0.0
            long_sum = 0.0
            dom_fee_pax_sum = 0.0
            dom_fee_mov_sum = [0.0] * n_aircraft
            intnl_fee_pax_sum = 0.0
            intnl_fee_mov_sum = [0.0] * n_aircraft
            taxi_out_sum = 0.0
            taxi_in_sum = 0.0
            airport_column = 1
            long_flag = False
            while f"Airport_{airport_column}" in city.index:
                airport_id = int(city[f"Airport_{airport_column}"])

                if not (airport_id == 0):
                    airport_row = airport_data.index[
                        airport_data["AirportID"] == airport_id
                    ].tolist()[0]

                    capacity_sum += float(
                        airport_data.at[airport_row, "Capacity_movts_hr"]
                    )

                    longest_runway_m = max(
                        longest_runway_m,
                        float(airport_data.at[airport_row, "RunwayLength_m"]),
                    )

                    lat_sum += float(airport_data.at[airport_row, "Latitude"]) * float(
                        airport_data.at[airport_row, "Capacity_movts_hr"]
                    )

                    # check for edge case where a city has airports either side of the 180deg longitude line
                    longitude = float(airport_data.at[airport_row, "Longitude"])
                    if longitude * long_sum < 0.0:
                        if longitude < 0.0:
                            longitude += 360.0
                        else:
                            longitude -= 360.0
                        long_flag = True
                    long_sum += longitude * float(
                        airport_data.at[airport_row, "Capacity_movts_hr"]
                    )

                    dom_fee_pax_sum += float(
                        airport_data.at[
                            airport_row, "LandingCosts_PerPax_Domestic_2015USdollars"
                        ]
                    )
                    dom_fee_mov_sum = list(
                        map(
                            add,
                            dom_fee_mov_sum,
                            [
                                float(
                                    airport_data.at[
                                        airport_row,
                                        f"LandingCosts_PerMovt_Size{ac}_Domestic_2015USdollars",
                                    ]
                                )
                                for ac in range(n_aircraft)
                            ],
                        )
                    )

                    intnl_fee_pax_sum += float(
                        airport_data.at[
                            airport_row,
                            "LandingCosts_PerPax_International_2015USdollars",
                        ]
                    )
                    intnl_fee_mov_sum = list(
                        map(
                            add,
                            intnl_fee_mov_sum,
                            [
                                float(
                                    airport_data.at[
                                        airport_row,
                                        f"LandingCosts_PerMovt_Size{ac}_International_2015USdollars",
                                    ]
                                )
                                for ac in range(n_aircraft)
                            ],
                        )
                    )

                    taxi_out_sum += float(
                        airport_data.at[airport_row, "UnimpTaxiOut_min"]
                    )
                    taxi_in_sum += float(
                        airport_data.at[airport_row, "UnimpTaxiIn_min"]
                    )
                else:
                    # airport_id == 0 => there are no more airports for that city
                    break
                airport_column += 1

            city_longitude = long_sum / capacity_sum
            if long_flag:
                if city_longitude < -180.0:
                    city_longitude += 360.0
                if city_longitude > 180.0:
                    city_longitude -= 360.0

            city_lookup[city["Country"]].append(city_id)

            cities[city_id] = City(
                city_id=city_id,
                city_name=city["CityName"],
                region=city["Region"],
                country=city["Country"],
                local_region=city["LocalRegion"],
                capital_city=city["CapitalCity"],
                base_population=city["BaseYearPopulation"],
                population=city["BaseYearPopulation"],
                base_income_USDpercap=city["BaseYearIncome"],
                income_USDpercap=city["BaseYearIncome"],
                latitude=lat_sum / capacity_sum,
                longitude=city_longitude,
                domestic_fees_USDperpax=dom_fee_pax_sum / capacity_sum,
                domestic_fees_USDpermvmt=list(
                    map(lambda x: x / capacity_sum, dom_fee_mov_sum)
                ),
                international_fees_USDperpax=intnl_fee_pax_sum / capacity_sum,
                international_fees_USDpermvmt=list(
                    map(lambda x: x / capacity_sum, intnl_fee_mov_sum)
                ),
                taxi_out_mins=taxi_out_sum / capacity_sum,
                taxi_in_mins=taxi_in_sum / capacity_sum,
                capacity_perhr=capacity_sum,
                longest_runway_m=longest_runway_m,
            )

        return cities, city_lookup
