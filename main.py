import os
import pandas as pd
from city import City
from airline import Airline
from route import Route
from aircraft import Aircraft
import random


def main():
    # import data
    file_path = os.path.dirname(__file__)
    run_data = pd.read_csv(os.path.join(file_path, "RunParameters.csv"))
    n_runs = run_data.shape[0]

    for run_idx, run_parameters in run_data.iterrows():
        print(f"\nSimulation {run_idx+1} of {n_runs}\n")
        print(f"RunNumber: {run_parameters['RunNumber']}")
        print(f"RunID:     {run_parameters['RunID']}\n")

        # create a pseudorandom number generator that generates repeatable 'randomness'
        randomGen = random.Random(x=0)

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

        # make adjustments to data format where relevant
        # change country code from 100+ to 0+ to allow for list indexing by code
        airport_data["CountryCode"] = airport_data["CountryCode"] - 100
        city_data["Country"] = city_data["Country"] - 100
        country_data["CountryCode"] = country_data["CountryCode"] - 100
        # calculate representative ranges for each aircraft type
        aircraft_data["TypicalRange_m"] = Aircraft.calc_ranges(aircraft_data, run_parameters["StartYear"])

        # sort data where relevant
        city_data.sort_values(by="CityID", inplace=True)
        fleet_data.sort_values(by="AircraftID", inplace=True)
        aircraft_data.sort_values(by="TypicalRange_m", inplace=True, ascending=False)

        # create classes
        print("    Initialising cities...")
        cities, city_lookup = City.initialise_cities(
            city_data=city_data,
            airport_data=airport_data,
        )

        print("    Initialising airlines...")
        airlines = Airline.initialise_airlines(
            fleet_data=fleet_data,
            country_data=country_data,
            run_parameters=run_parameters,
        )

        print("    Initialising routes...")
        routes = Route.initialise_routes(
            cities=cities,
            city_pair_data=city_pair_data,
            price_elasticities=price_elasticities,
            income_elasticities=income_elasticities,
            population_elasticity=run_parameters["PopulationElasticity"],
        )

        # initialise airline fleet assignment
        for airline in airlines:
            airline.initialise_fleet_assignment(routes, aircraft_data, city_lookup, randomGen, run_parameters["StartYear"])

        # simulate base year


        # derive correction factors


        # iterate over desired years



    return 0


if __name__ == "__main__":
    main()
