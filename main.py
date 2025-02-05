import os
import pandas as pd
import initialisation


def main():
    # import data
    file_path = os.path.dirname(__file__)
    run_data = pd.read_csv(os.path.join(file_path, "RunParameters.csv"))
    n_runs = run_data.shape[0]

    for run_idx, run_parameters in run_data.iterrows():
        print(f"\nSimulation {run_idx+1} of {n_runs}\n")
        print(f"RunNumber: {run_parameters['RunNumber']}")
        print(f"RunID:     {run_parameters['RunID']}\n")

        data_path = os.path.join(file_path, str(run_parameters["DataInputDirectory"]))
        save_folder_path = os.path.join(file_path, "output")
        if not os.path.exists(save_folder_path):
            os.makedirs(save_folder_path)
        savepath = os.path.join(
            save_folder_path, f"output_{run_parameters["RunID"]}.csv"
        )

        # import run-specific data
        print(f"    Importing data from {data_path}")

        aircraft_data = pd.read_csv(os.path.join(data_path, "AircraftData.csv"))
        airport_data = pd.read_csv(os.path.join(data_path, "AirportData.csv"))
        city_data = pd.read_csv(os.path.join(data_path, "CityData.csv"))
        country_data = pd.read_csv(os.path.join(data_path, "CountryData.csv"))
        fleet_data = pd.read_csv(
            os.path.join(data_path, "FleetDataPassenger.csv")
        )  # note cargo aircraft are in a seperate file
        city_pair_data = pd.read_csv(os.path.join(data_path, "DataByCityPair.csv"))
        price_elasticities = pd.read_csv(os.path.join(data_path, "PriceElasticities.csv"))
        income_elasticities = pd.read_csv(os.path.join(data_path, "IncomeElasticities.csv"))

        # create classes
        print("    Initialising cities...")
        city_data.sort_values(by="CityID", inplace=True)
        cities = initialisation.initialise_cities(city_data, airport_data)
        print("    Initialising airlines...")
        airlines = initialisation.initialise_airlines(
            fleet_data, country_data, run_parameters
        )
        print("    Initialising routes...")
        routes = initialisation.initialise_routes(cities, city_pair_data, price_elasticities, income_elasticities)

        # simulate base year


        # derive correction factors


        # iterate over desired years



    return 0


if __name__ == "__main__":
    main()
