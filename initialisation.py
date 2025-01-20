import pandas as pd
from operator import add
import classes


def initialise_cities(city_data: pd.DataFrame, airport_data: pd.DataFrame) -> list:
    """
    Generate list of instances of City dataclass from contents of CityData and AirportData files

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
    while (f"LandingCosts_PerMovt_Size{n_aircraft}_Domestic_2015USdollars" in airport_data.columns):
        n_aircraft += 1

    cities = []
    # loop over rows of city_data
    for idx, city in city_data.iterrows():
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
        while (f"Airport_{airport_column}" in city.index):
            airport_id = int(city[f"Airport_{airport_column}"])

            if not (airport_id == 0):
                airport_row = airport_data.index[airport_data["AirportID"] == airport_id].tolist()[0]

                capacity_sum += float(airport_data.at[airport_row, "Capacity_movts_hr"])

                lat_sum += (float(airport_data.at[airport_row, "Latitude"]) * float(airport_data.at[airport_row, "Capacity_movts_hr"]))

                # check for edge case where a city has airports either side of the 180deg longitude line
                longitude = float(airport_data.at[airport_row, "Longitude"])
                if longitude * long_sum < 0.0:
                    if longitude < 0.0:
                        longitude += 360.0
                    else:
                        longitude -= 360.0
                    long_flag = True
                long_sum += (longitude * float(airport_data.at[airport_row, "Capacity_movts_hr"]))

                dom_fee_pax_sum += float(airport_data.at[airport_row, "LandingCosts_PerPax_Domestic_2015USdollars"])
                dom_fee_mov_sum = list(map(add, dom_fee_mov_sum, [float(airport_data.at[airport_row, f"LandingCosts_PerMovt_Size{ac}_Domestic_2015USdollars"]) for ac in range(n_aircraft)]))

                intnl_fee_pax_sum += float(airport_data.at[airport_row, "LandingCosts_PerPax_International_2015USdollars"])
                intnl_fee_mov_sum = list(map(add, intnl_fee_mov_sum, [float(airport_data.at[airport_row, f"LandingCosts_PerMovt_Size{ac}_International_2015USdollars"]) for ac in range(n_aircraft)]))

                taxi_out_sum += float(airport_data.at[airport_row, "UnimpTaxiOut_min"])
                taxi_in_sum += float(airport_data.at[airport_row, "UnimpTaxiIn_min"])
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

        cities.append(
            classes.City(
                city_id = city["CityID"],
                city_name = city["CityName"],
                region = city["Region"],
                country = city["Country"],
                local_region = city["LocalRegion"],
                capital_city = city["CapitalCity"],
                population = city["BaseYearPopulation"],
                income_USDpercap = city["BaseYearIncome"],
                latitude = lat_sum / capacity_sum,
                longitude = city_longitude,
                domestic_fees_USDperpax = dom_fee_pax_sum / capacity_sum,
                domestic_fees_USDpermvmt = list(map(lambda x: x/capacity_sum, dom_fee_mov_sum)),
                international_fees_USDperpax = intnl_fee_pax_sum / capacity_sum,
                international_fees_USDpermvmt = list(map(lambda x: x/capacity_sum, intnl_fee_mov_sum)),
                taxi_out_mins = taxi_out_sum / capacity_sum,
                taxi_in_mins = taxi_in_sum / capacity_sum,
                capacity_perhr = capacity_sum,
            )
        )

    return cities


def initialise_airlines(fleet_data: pd.DataFrame, country_data: pd.DataFrame, run_parameters: pd.DataFrame) -> list:
    """
    Generate list of instances of Airline dataclass from contents of FleetData and CountryData files
    Within each region, the number of aircraft of each type assigned to each country is proportional to the country's GDP
    Within each country, the aircraft of each type are assigned to the airlines as evenly as possible

    Parameters
    ----------
    fleet_data : pd.DataFrame
    country_data : pd.DataFrame
    run_parameters : pd.DataFrame

    Returns
    -------
    airlines : list of instances of Airline dataclass
    """
    census_regions = {
        'Americas': [10, 11, 12],
        'Europe': [13],
        'MiddleEast': [14],
        'Africa': [15],
        'AsiaPacific': [16]
    }

    country_data['GDP'] = country_data['PPP_GDP_Cap_Year2015USD_2015'] * country_data['Population_2015']
    country_data['Region_GDP_Proportion'] = 0.0  # initialise with zeros

    airlines = []
    airline_idx = 0
    n_aircraft_types = len(fleet_data)

    for region, codes in census_regions.items():
        # calculate GDP proportions
        countries = country_data['Region'].isin(codes)
        if not any(countries):
            continue
        region_gdp = country_data.loc[countries, 'GDP'].sum()
        country_data.loc[countries, 'Region_GDP_Proportion'] = (
            country_data.loc[countries, 'GDP'] / region_gdp
        )

        # sort countries in region by GDP
        sorted_region = sorted(
                country_data.loc[countries].iterrows(),
                key=lambda x: (x[1]['GDP'], x[1]['Population_2015']),  # Use population as tiebreaker
                reverse=True
            )

        # assign aircraft to countries
        region_aircraft = fleet_data[f"Census{region}"].to_list()
        unallocated = region_aircraft.copy()
        for country_idx, country in sorted_region:  # no need for iterrows due to sorting method
            country_aircraft = [0] * n_aircraft_types
            # iterate through aircraft types and account for rounding issues
            for i in range(n_aircraft_types):
                allocated = int(round(region_aircraft[i] * country['Region_GDP_Proportion']))
                if allocated == 0 and unallocated[i] > 0:
                    allocated = 1
                country_aircraft[i] = min(allocated, unallocated[i])
                unallocated[i] -= country_aircraft[i]

            # assign aircraft to airlines within country
            for country_airline_idx in range(run_parameters["AirlinesPerCountry"]):
                n_aircraft = [0] * n_aircraft_types
                for aircraft_type in range(n_aircraft_types):
                    base = country_aircraft[aircraft_type] // run_parameters["AirlinesPerCountry"]
                    remainder = country_aircraft[aircraft_type] % run_parameters["AirlinesPerCountry"]
                    # add one extra aircraft to each airline until the remainder is used up
                    n_aircraft[aircraft_type] = base + (1 if country_airline_idx < remainder else 0)

                # create airline, unless it has no aircraft
                if sum(n_aircraft) > 0:
                    airlines.append(
                        classes.Airline(
                            airline_id = airline_idx,
                            region = country['Region'],
                            country = country['Country'],
                            country_num = country['Number'],
                            n_aircraft = n_aircraft
                        )
                    )

                    airline_idx += 1
    return airlines
