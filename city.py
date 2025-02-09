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

    @staticmethod
    def initialise_cities(city_data: pd.DataFrame, airport_data: pd.DataFrame) -> list:
        """
        Generate list of instances of City dataclass from contents of CityData and AirportData files
        List is indexed to match CityID field in CityData file
        If a certain index doesn't exist as a CityID in CityData, the corresponding list element is None

        Parameters
        ----------
        city_data : pd.DataFrame
        airport_data : pd.DataFrame

        Returns
        -------
        cities : list of instances of City dataclass
        """
        # check number of different aircraft supported by airport_data
        n_aircraft = 0
        while (
            f"LandingCosts_PerMovt_Size{n_aircraft}_Domestic_2015USdollars"
            in airport_data.columns
        ):
            n_aircraft += 1

        # find CityID of last city in city_data
        last_city_id = city_data["CityID"].iloc[-1]
        cities = [None] * (last_city_id + 1)

        # loop over rows of city_data
        # for idx, city in city_data.iterrows():
        for city_id in range(last_city_id + 1):
            # check whether city_id is a CityID in city_data
            if not (city_id in city_data["CityID"].values):
                continue  # leave cities[city_id] as None

            city = city_data.loc[city_data["CityID"] == city_id]
            city = city.squeeze()  # convert from DataFrame to Series

            # calculate capacity-weighted mean of airport data
            capacity_sum = 0.0
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
            )

        return cities
