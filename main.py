import os
import pandas as pd
import city
import route
import airline
import aircraft
import simulator
import random
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

        # create list of regions to simulate
        if (
            run_parameters["Regions"] == "all"
            or run_parameters["Regions"] == ""
            or run_parameters["Regions"] == "[]"
        ):
            regions = None  # flag to run all regions
        else:
            regions = [int(i.strip()) for i in run_parameters["Regions"].strip("[]").split(",")]

        if regions is None:
            print("    Running global simulation...")
        else:
            print(f"    Running simulation for regions: {regions}")

        data_path = os.path.join(file_path, str(run_parameters["DataInputDirectory"]))
        save_folder_path = os.path.join(file_path, f"output_{run_parameters['RunID']}")
        if not os.path.exists(save_folder_path):
            os.makedirs(save_folder_path)

        # import cached data if relevant
        if run_parameters["CacheOption"] not in ["load", "save", "none"]:
            raise ValueError(f"Invalid CacheOption: {run_parameters['CacheOption']}")
        if run_parameters["CacheOption"] in ["load", "save"]:
            cache_folder_path = os.path.join(file_path, run_parameters["CacheDirectory"])
            if not os.path.exists(cache_folder_path):
                os.makedirs(cache_folder_path)
        city_data_cache = False
        city_lookup_cache = False
        airlines_cache = False
        city_pair_data_cache = False
        airline_fleets_cache = False
        airline_routes_cache = False
        if run_parameters["CacheOption"] == "load":
            # load from cache
            try:
                with open(os.path.join(cache_folder_path, "city_data.pkl"), "rb") as f:
                    city_data = pickle.load(f)
                    city_data_cache = True
            except FileNotFoundError:
                print("    city_data cache load failed. Reinitialising data...")
            try:
                with open(os.path.join(cache_folder_path, "city_lookup.pkl"), "rb") as f:
                    city_lookup = pickle.load(f)
                    city_lookup_cache = True
            except FileNotFoundError:
                print("    city_lookup cache load failed. Reinitialising data...")
            try:
                with open(os.path.join(cache_folder_path, "airlines.pkl"), "rb") as f:
                    airlines = pickle.load(f)
                    airlines_cache = True
            except FileNotFoundError:
                print("    airlines cache load failed. Reinitialising data...")
            try:
                with open(os.path.join(cache_folder_path, "city_pair_data.pkl"), "rb") as f:
                    city_pair_data = pickle.load(f)
                    city_pair_data_cache = True
            except FileNotFoundError:
                print("    city_pair_data cache load failed. Reinitialising data...")
            try:
                with open(os.path.join(cache_folder_path, "airline_fleets.pkl"), "rb") as f:
                    airline_fleets = pickle.load(f)
                    airline_fleets_cache = True
            except FileNotFoundError:
                print("    airline_fleets cache load failed. Reinitialising data...")
            try:
                with open(os.path.join(cache_folder_path, "airline_routes.pkl"), "rb") as f:
                    airline_routes = pickle.load(f)
                    airline_routes_cache = True
            except FileNotFoundError:
                print("    airline_routes cache load failed. Reinitialising data...")

        # import, sort and initialise run-specific data if not loaded from cache
        print(f"    Importing data from {data_path}")

        aircraft_data = pd.read_csv(os.path.join(data_path, "AircraftData.csv"))
        aircraft_data.set_index('AircraftID', inplace=True)
        aircraft_data["TypicalRange_m"] = aircraft.calc_ranges(aircraft_data, run_parameters["StartYear"])
        aircraft_data.sort_values(by="TypicalRange_m", inplace=True, ascending=False)

        airport_data = pd.read_csv(os.path.join(data_path, "AirportData.csv"))
        # change country code from 100+ to 0+ to allow for list indexing by code
        airport_data["Country"] = airport_data["Country"] - 100
        airport_data.sort_values(by="AirportID", inplace=True)

        country_data = pd.read_csv(os.path.join(data_path, "CountryData.csv"))
        # change country code from 100+ to 0+ to allow for list indexing by code
        country_data["Number"] = country_data["Number"] - 100

        population_data = pd.read_csv(os.path.join(data_path, "PopulationProjections.csv"))
        # change country code from 100+ to 0+ to allow for list indexing by code
        population_data["Number"] = population_data["Number"] - 100
        income_data = pd.read_csv(os.path.join(data_path, "IncomeProjections.csv"))
        # change country code from 100+ to 0+ to allow for list indexing by code
        income_data["Number"] = income_data["Number"] - 100
        country_data = city.set_base_values(
            country_data,
            population_data,
            income_data,
            run_parameters["StartYear"],
        )

        price_elasticities = pd.read_csv(os.path.join(data_path, "PriceElasticities.csv"))
        income_elasticities = pd.read_csv(os.path.join(data_path, "IncomeElasticities.csv"))
        fuel_data = pd.read_csv(os.path.join(data_path, "FuelProjection.csv"))
        with open(os.path.join(data_path, "DemandCoefficients.csv"), "r", encoding='utf-8-sig') as f:
            demand_coefficients = dict(zip(f.readline().strip().split(","), map(float, f.readline().strip().split(","))))
        
        airport_expansion_data = pd.read_csv(os.path.join(data_path, "AirportExpansion.csv"))
        
        fleet_data = pd.read_csv(
            os.path.join(data_path, "FleetDataPassenger.csv")
        )  # note cargo aircraft are in a seperate file
        fleet_data.sort_values(by="AircraftID", inplace=True)
        
        if not city_data_cache or not city_lookup_cache:
            city_data = pd.read_csv(os.path.join(data_path, "CityData.csv"))
            # change country code from 100+ to 0+ to allow for list indexing by code
            city_data["Country"] = city_data["Country"] - 100
            city_data.sort_values(by="CityID", inplace=True)

            print("    Initialising cities...")
            city_data, city_lookup = city.add_airports_to_cities(
                city_data, airport_data
            )
            if run_parameters["CacheOption"] == "save":
                with open(os.path.join(cache_folder_path, "city_lookup.pkl"), "wb") as f:
                    pickle.dump(city_lookup, f)

        if not airlines_cache:
            print("    Initialising airlines...")
            airlines = airline.initialise_airlines(
                fleet_data,
                country_data,
                run_parameters,
                regions,
            )

        if not city_pair_data_cache:
            print("    Initialising routes...")
            city_pair_data = pd.read_csv(os.path.join(data_path, "DataByCityPair.csv"))
            city_pair_data = route.initialise_routes(
                city_data,
                city_pair_data,
                price_elasticities,
                income_elasticities,
                run_parameters["PopulationElasticity"],
            )

        if (
            not airline_fleets_cache
            or not airline_routes_cache
            or not city_pair_data_cache
            or not city_data_cache
            or not airlines_cache
        ):
            run_parameters["RerunFareInit"] = "y"
            print("    Initialising fleet assignment...")
            (
                airline_fleets,
                airline_routes,
                city_pair_data,
                city_data,
                capacity_flag_list,
            ) = (
                airline.initialise_fleet_assignment(
                    airlines,
                    city_pair_data,
                    city_data,
                    aircraft_data,
                    city_lookup,
                    randomGen,
                    run_parameters["StartYear"],
                    demand_coefficients,
                    regions,
                )
            )

            airlines, city_pair_data, city_data, country_data = simulator.limit_to_region(
                regions,
                airlines,
                city_pair_data,
                city_data,
                country_data,
            )

            print("    Checking airport capacity limits...")
            if len(capacity_flag_list) > 0:
                print(f"        Limits exceeded for {len(capacity_flag_list)} city/cities. Reassigning fleets...")
                FuelCost_USDperGallon = fuel_data.loc[
                    fuel_data["Year"] == run_parameters["StartYear"], "Price_USD_per_Gallon"
                ].values[0]
                (
                    airline_fleets,
                    airline_routes,
                    city_pair_data,
                    city_data,
                    airlines,
                ) = city.enforce_capacity(
                    airlines,
                    airline_fleets,
                    airline_routes,
                    aircraft_data,
                    city_pair_data,
                    city_data,
                    city_lookup,
                    capacity_flag_list,
                    demand_coefficients,
                    FuelCost_USDperGallon,
                )
            else:
                print("    No capacity limits exceeded.")

        if run_parameters["CacheOption"] == "save":
            with open(os.path.join(cache_folder_path, "city_pair_data.pkl"), "wb") as f:
                pickle.dump(city_pair_data, f)
            with open(os.path.join(cache_folder_path, "city_data.pkl"), "wb") as f:
                pickle.dump(city_data, f)
            with open(os.path.join(cache_folder_path, "airline_fleets.pkl"), "wb") as f:
                pickle.dump(airline_fleets, f)
            with open(os.path.join(cache_folder_path, "airline_routes.pkl"), "wb") as f:
                pickle.dump(airline_routes, f)
            with open(os.path.join(cache_folder_path, "airlines.pkl"), "wb") as f:
                pickle.dump(airlines, f)

        country_data = country_data[country_data["Region"].isin(regions)]  # needs to be done every time since country_data is not saved to pkl
        
        # run simulation
        simulator.run_simulation(
            airlines,
            airline_fleets,
            airline_routes,
            city_data,
            city_pair_data,
            city_lookup,
            country_data,
            aircraft_data,
            demand_coefficients,
            run_parameters["PopulationElasticity"],
            population_data,
            income_data,
            fuel_data,
            save_folder_path,
            cache_folder_path,
            airport_expansion_data,
            run_parameters,
        )

    return 0


if __name__ == "__main__":
    main()
