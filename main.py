import os
import pandas as pd
import classes
from operator import add


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
                income_pc = city["BaseYearIncome"],
                latitude = lat_sum / capacity_sum,
                longitude = city_longitude,
                domestic_fees_perpax = dom_fee_pax_sum / capacity_sum,
                domestic_fees_permvmt = list(map(lambda x: x/capacity_sum, dom_fee_mov_sum)),
                international_fees_perpax = intnl_fee_pax_sum / capacity_sum,
                international_fees_permvmt = list(map(lambda x: x/capacity_sum, intnl_fee_mov_sum)),
                taxi_out_mins = taxi_out_sum / capacity_sum,
                taxi_in_mins = taxi_in_sum / capacity_sum,
                capacity_perhr = capacity_sum,
            )
        )

    return cities







def main():
    # import data
    file_path = os.path.dirname(__file__)
    run_parameters = pd.read_csv(os.path.join(file_path, "RunParameters.csv"))

    data_path = os.path.join(file_path, str(run_parameters["DataInputDirectory"][0]))
    save_folder_path = os.path.join(file_path, "output")
    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)
    savepath = os.path.join(
        save_folder_path, "output_" + str(run_parameters["RunID"][0]) + ".csv"
    )

    aircraft_data = pd.read_csv(os.path.join(data_path, "AircraftData.csv"))
    airport_data = pd.read_csv(os.path.join(data_path, "AirportData.csv"))
    city_data = pd.read_csv(os.path.join(data_path, "CityData.csv"))

    # create classes
    cities = initialise_cities(city_data, airport_data)

    # simulate base year


    # derive correction factors


    # iterate over desired years



    return 0


if __name__ == "__main__":
    main()