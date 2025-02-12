import os
import pandas as pd
import city
from airline import Airline
from route import Route
from aircraft import Aircraft
import random
from tqdm import tqdm
import pickle


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
        airport_data["Country"] = airport_data["Country"] - 100
        city_data["Country"] = city_data["Country"] - 100
        country_data["Number"] = country_data["Number"] - 100
        # calculate representative ranges for each aircraft type
        aircraft_data["TypicalRange_m"] = Aircraft.calc_ranges(aircraft_data, run_parameters["StartYear"])

        # sort data where relevant
        city_data.sort_values(by="CityID", inplace=True)
        fleet_data.sort_values(by="AircraftID", inplace=True)
        aircraft_data.sort_values(by="TypicalRange_m", inplace=True, ascending=False)

        # create classes or load from cache
        if run_parameters["CacheOption"] not in ["load", "save", "none"]:
            raise ValueError(f"Invalid CacheOption: {run_parameters['CacheOption']}")
        initialised_from_cache = False
        if run_parameters["CacheOption"] == "load":
            # load from cache
            cache_folder_path = os.path.join(file_path, run_parameters["CacheDirectory"])
            use_cache = input(f"    Loading initialisation data from {cache_folder_path}. Only continue if you trust data in this folder. Continue? (y/n): ").lower()
            while use_cache not in ["y", "n"]:
                use_cache = input("    Please enter 'y' or 'n': ").lower()
            if use_cache == "y":
                try:
                    with open(os.path.join(cache_folder_path, "cities.pkl"), "rb") as f:
                        cities = pickle.load(f)
                    with open(os.path.join(cache_folder_path, "airlines.pkl"), "rb") as f:
                        airlines = pickle.load(f)
                    with open(os.path.join(cache_folder_path, "routes.pkl"), "rb") as f:
                        routes = pickle.load(f)
                    with open(os.path.join(cache_folder_path, "city_lookup.pkl"), "rb") as f:
                        city_lookup = pickle.load(f)
                    initialised_from_cache = True
                except FileNotFoundError:
                    print("    Cache load failed. Reinitialising data...")
            else:
                print("    Cache load cancelled. Reinitialising data...")

        if not initialised_from_cache:
            # initialise classes
            print("    Initialising cities...")
            city_data, city_lookup = city.add_airports_to_cities(
                city_data, airport_data
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
            for airline in tqdm(
                airlines,
                desc="        Fleet assignment",
                ascii=False,
                ncols=75,
            ):
                airline.initialise_fleet_assignment(
                    routes,
                    cities,
                    aircraft_data,
                    city_lookup,
                    randomGen,
                    run_parameters["StartYear"]
                )

            # save initialisation data to cache
            if run_parameters["CacheOption"] == "save":
                cache_folder_path = os.path.join(file_path, run_parameters["CacheDirectory"])
                print(f"    Saving initialisation data to {cache_folder_path}")
                if not os.path.exists(cache_folder_path):
                    os.makedirs(cache_folder_path)
                with open(os.path.join(cache_folder_path, "cities.pkl"), "wb") as f:
                    pickle.dump(cities, f)
                with open(os.path.join(cache_folder_path, "airlines.pkl"), "wb") as f:
                    pickle.dump(airlines, f)
                with open(os.path.join(cache_folder_path, "routes.pkl"), "wb") as f:
                    pickle.dump(routes, f)
                with open(os.path.join(cache_folder_path, "city_lookup.pkl"), "wb") as f:
                    pickle.dump(city_lookup, f)
        
        # simulate base year


        # derive correction factors


        # iterate over desired years



    return 0


if __name__ == "__main__":
    main()
