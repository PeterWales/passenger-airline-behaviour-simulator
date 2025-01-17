import os
import pandas as pd
import initialisation



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
    cities = initialisation.initialise_cities(city_data, airport_data)

    # simulate base year


    # derive correction factors


    # iterate over desired years



    return 0


if __name__ == "__main__":
    main()
    