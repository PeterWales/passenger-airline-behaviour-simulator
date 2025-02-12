import pandas as pd
from operator import add


def add_airports_to_cities(city_data: pd.DataFrame, airport_data: pd.DataFrame) -> tuple[pd.DataFrame, list[list]]:
    """
    Incorporate parts of airport_data into city_data DataFrame so that cities can be treated as single entities.
    Generate a list of lists for looking up which cities are located in which countries.

    cities list is indexed to match CityID field in CityData file
    If a certain index doesn't exist as a CityID in CityData, the corresponding list element is None
    city_lookup list is indexed to match CountryID field in CountryData file, and contains lists of CityIDs
    If a certain country doesn't contain any cities, the corresponding list element is empty

    Parameters
    ----------
    city_data : pd.DataFrame
    airport_data : pd.DataFrame

    Returns
    -------
    city_data : pd.DataFrame
    city_lookup : list[list]
    """
    # TODO: use a package for geodesic calculations

    # check number of different aircraft supported by airport_data
    n_aircraft = 0
    while (
        f"LandingCosts_PerMovt_Size{n_aircraft}_Domestic_2015USdollars"
        in airport_data.columns
    ):
        n_aircraft += 1
    if n_aircraft == 0:
        raise ValueError("No aircraft-specific data found in airport_data")

    # initialise lists
    last_country_id = max(city_data["Country"])
    city_lookup = [[] for _ in range(last_country_id + 1)]
    capacity = []
    latitude = []
    longitude = []
    domestic_fees_USDperpax = []
    domestic_fees_USDpermvmt = []
    international_fees_USDperpax = []
    international_fees_USDpermvmt = []
    taxi_out_mins = []
    taxi_in_mins = []
    capacity_perhr = []
    longest_runway_m = []

    # loop over rows of city_data
    for idx, city in city_data.iterrows():
        # calculate capacity-weighted mean of airport data
        capacity_sum = 0.0
        longest_runway_found = 0.0
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

                longest_runway_found = max(
                    longest_runway_found,
                    float(airport_data.at[airport_row, "LongestRunway_m"]),
                )

                lat_sum += float(airport_data.at[airport_row, "Latitude"]) * float(
                    airport_data.at[airport_row, "Capacity_movts_hr"]
                )

                # check for edge case where a city has airports either side of the 180deg longitude line
                apt_longitude = float(airport_data.at[airport_row, "Longitude"])
                if apt_longitude * long_sum < 0.0:
                    if apt_longitude < 0.0:
                        apt_longitude += 360.0
                    else:
                        apt_longitude -= 360.0
                    long_flag = True
                long_sum += apt_longitude * float(
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

        city_lookup[city["Country"]].append(city["CityID"])

        capacity.append(capacity_sum)
        latitude.append(lat_sum / capacity_sum)
        longitude.append(city_longitude)
        domestic_fees_USDperpax.append(dom_fee_pax_sum / capacity_sum)
        domestic_fees_USDpermvmt.append(
            list(map(lambda x: x / capacity_sum, dom_fee_mov_sum))
        )
        international_fees_USDperpax.append(intnl_fee_pax_sum / capacity_sum)
        international_fees_USDpermvmt.append(
            list(map(lambda x: x / capacity_sum, intnl_fee_mov_sum))
        )
        taxi_out_mins.append(taxi_out_sum / capacity_sum)
        taxi_in_mins.append(taxi_in_sum / capacity_sum)
        capacity_perhr.append(capacity_sum)
        longest_runway_m.append(longest_runway_found)

    # add new columns to city_data

    city_data["Population"] = city_data["BaseYearPopulation"]
    city_data["Income_USDpercap"] = city_data["BaseYearIncome"]
    city_data["Capacity_MovtsPerHr"] = capacity
    city_data["Latitude"] = latitude
    city_data["Longitude"] = longitude
    city_data["Domestic_Fees_USDperPax"] = domestic_fees_USDperpax
    city_data["Domestic_Fees_USDperMovt"] = domestic_fees_USDpermvmt
    city_data["International_Fees_USDperPax"] = international_fees_USDperpax
    city_data["International_Fees_USDperMovt"] = international_fees_USDpermvmt
    city_data["Taxi_Out_mins"] = taxi_out_mins
    city_data["Taxi_In_mins"] = taxi_in_mins
    city_data["Capacity_PerHr"] = capacity_perhr
    city_data["LongestRunway_m"] = longest_runway_m

    return city_data, city_lookup
