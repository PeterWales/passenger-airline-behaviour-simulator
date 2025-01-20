import os
import pandas as pd
import initialisation



def main():
    # import data
    file_path = os.path.dirname(__file__)
    run_data = pd.read_csv(os.path.join(file_path, "RunParameters.csv"))

    for run_idx, run_parameters in run_data.iterrows(): 
        data_path = os.path.join(file_path, str(run_parameters["DataInputDirectory"]))
        save_folder_path = os.path.join(file_path, "output")
        if not os.path.exists(save_folder_path):
            os.makedirs(save_folder_path)
        savepath = os.path.join(
            save_folder_path, f"output_{run_parameters["RunID"]}.csv"
        )

        aircraft_data = pd.read_csv(os.path.join(data_path, "AircraftData.csv"))
        airport_data = pd.read_csv(os.path.join(data_path, "AirportData.csv"))
        city_data = pd.read_csv(os.path.join(data_path, "CityData.csv"))
        country_data = pd.read_csv(os.path.join(data_path, "CountryData.csv"))
        fleet_data = pd.read_csv(os.path.join(data_path, "FleetDataPassenger.csv")) # note doesn't include cargo aircraft

        # create classes
        cities = initialisation.initialise_cities(city_data, airport_data)
        airlines = initialisation.initialise_airlines(fleet_data, country_data, run_parameters)

        # simulate base year


        # derive correction factors


        # iterate over desired years



    return 0


if __name__ == "__main__":
    main()
    