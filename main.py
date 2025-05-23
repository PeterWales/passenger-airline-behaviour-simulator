import os
import pandas as pd
import city
import route
import airline
import aircraft
import simulator
import random
import pickle
import datetime
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='Airline Schedule Predictor')
    parser.add_argument(
        '--paramfile', 
        type=str, 
        default='RunParameters.csv',
        help='Path to the CSV file containing run parameters (default: RunParameters.csv)'
    )
    return parser.parse_args()


def main():
    # parse command line arguments
    args = parse_arguments()

    # import data
    file_path = os.path.dirname(__file__)
    run_parameters = pd.read_csv(os.path.join(file_path, args.paramfile))
    run_parameters = run_parameters.iloc[0]

    print(f"\nRunning simulation from file: {args.paramfile}\n")
    print(f"RunID: {run_parameters['RunID']}")
    print("Time started: ", datetime.datetime.now(), "\n")

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
    cache_folder_path = os.path.join(file_path, run_parameters["CacheDirectory"])
    if not os.path.exists(cache_folder_path):
        os.makedirs(cache_folder_path)

    # import data that doesn't get cached
    income_data = pd.read_csv(os.path.join(data_path, "IncomeProjections.csv"))
    population_data = pd.read_csv(os.path.join(data_path, "PopulationProjections.csv"))
    price_elasticities = pd.read_csv(os.path.join(data_path, "PriceElasticities.csv"))
    income_elasticities = pd.read_csv(os.path.join(data_path, "IncomeElasticities.csv"))
    conv_fuel_data = pd.read_csv(os.path.join(data_path, "FuelProjection.csv"))
    with open(os.path.join(data_path, "DemandCoefficients.csv"), "r", encoding='utf-8-sig') as f:
        demand_coefficients = dict(zip(f.readline().strip().split(","), map(float, f.readline().strip().split(","))))
    airport_expansion_data = pd.read_csv(os.path.join(data_path, "AirportExpansion.csv"))
    saf_mandate_data = pd.read_csv(os.path.join(data_path, "SAFMandate.csv"))
    saf_pathway_data = pd.read_csv(os.path.join(data_path, "SAFPathways.csv"))
    fleet_data = pd.read_csv(
        os.path.join(data_path, "FleetDataPassenger.csv")
    )  # note cargo aircraft are in a seperate file
    fleet_data.sort_values(by="AircraftID", inplace=True)

    # change country codes from 100+ to 0+ to allow for list indexing by code
    income_data["Number"] = income_data["Number"] - 100
    population_data["Number"] = population_data["Number"] - 100
    
    if run_parameters["ContinueExistingSim"] == "n" or run_parameters["ContinueExistingSim"] == "N":
        # import cached data if relevant
        if run_parameters["CacheOption"] not in ["load", "save", "none"]:
            raise ValueError(f"Invalid CacheOption: {run_parameters['CacheOption']}")
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

        country_data = city.set_base_values(
            country_data,
            population_data,
            income_data,
            run_parameters["StartYear"],
        )
        
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
                airlines,
            ) = (
                airline.initialise_fleet_assignment(
                    airlines,
                    city_pair_data,
                    city_data,
                    aircraft_data,
                    city_lookup,
                    random.Random(x=0),
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
                CJFCost_USDperGallon = conv_fuel_data.loc[
                    conv_fuel_data["Year"] == run_parameters["StartYear"], "Price_USD_per_Gallon"
                ].values[0]
                # neglect SAF cost for this
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
                    CJFCost_USDperGallon,
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

        # limit to the most popular routes to run faster (don't include this in pkl to allow this to be changed for each sim)
        city_pair_data = route.limit_routes(
            city_pair_data,
            run_parameters["RouteProportion"],
        )
    else:
        print(f"    Continuing existing simulation from RunID: {run_parameters['IDToContinue']}...")
        # load from intermediate cache
        annual_cache_path = os.path.join(cache_folder_path, f"intermediate_{run_parameters['IDToContinue']}")

        with open(os.path.join(annual_cache_path, "year_completed.pkl"), "rb") as f:
            year_completed = pickle.load(f)
        if run_parameters["YearToContinue"] > year_completed + 1:
            suffix = year_completed
            print(f"WARNING: YearToContinue is greater than year_completed+1. Starting from year_completed+1 ({year_completed+1}) instead.")
        else:
            suffix = run_parameters['YearToContinue'] - 1
        
        with open(os.path.join(annual_cache_path, f"airlines_{suffix}.pkl"), "rb") as f:
            airlines = pickle.load(f)
        with open(os.path.join(annual_cache_path, f"airline_fleets_{suffix}.pkl"), "rb") as f:
            airline_fleets = pickle.load(f)
        with open(os.path.join(annual_cache_path, f"airline_routes_{suffix}.pkl"), "rb") as f:
            airline_routes = pickle.load(f)
        with open(os.path.join(annual_cache_path, f"city_data_{suffix}.pkl"), "rb") as f:
            city_data = pickle.load(f)
        with open(os.path.join(annual_cache_path, f"city_pair_data_{suffix}.pkl"), "rb") as f:
            city_pair_data = pickle.load(f)
        with open(os.path.join(annual_cache_path, "city_lookup.pkl"), "rb") as f:
            city_lookup = pickle.load(f)
        with open(os.path.join(annual_cache_path, "country_data.pkl"), "rb") as f:
            country_data = pickle.load(f)
        with open(os.path.join(annual_cache_path, "aircraft_data.pkl"), "rb") as f:
            aircraft_data = pickle.load(f)
    
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
        conv_fuel_data,
        save_folder_path,
        cache_folder_path,
        airport_expansion_data,
        saf_mandate_data,
        saf_pathway_data,
        run_parameters,
    )

    return 0


if __name__ == "__main__":
    main()
