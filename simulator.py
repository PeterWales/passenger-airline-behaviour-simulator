import pandas as pd
import airline
import aircraft
import city
import os


def simulate_base_year(
    year: int,
    city_data: pd.DataFrame,
    city_pair_data: pd.DataFrame,
    airlines: pd.DataFrame,
    airline_fleets: list[pd.DataFrame],
    airline_routes: list[pd.DataFrame],
    aircraft_data: pd.DataFrame,
    FuelCost_USDperGallon: float,
    save_folder_path: str,
    max_fare: float,
):
    print(f"    Simulating base year ({year})...")

    # parameters for finding Nash equilibrium
    maxiters = 10
    demand_tolerance = 1000.0

    # initialise dataframes tracking convergence
    fare_iters = pd.DataFrame()
    fare_iters["Origin"] = city_pair_data["OriginCityID"]
    fare_iters["Destination"] = city_pair_data["DestinationCityID"]
    fare_iters["Origin_Latitude"] = city_pair_data["Origin_Latitude"]
    fare_iters["Origin_Longitude"] = city_pair_data["Origin_Longitude"]
    fare_iters["Destination_Latitude"] = city_pair_data["Destination_Latitude"]
    fare_iters["Destination_Longitude"] = city_pair_data["Destination_Longitude"]
    fare_iters["base"] = city_pair_data["Mean_Fare_USD"]

    demand_iters = pd.DataFrame()
    demand_iters["Origin"] = city_pair_data["OriginCityID"]
    demand_iters["Destination"] = city_pair_data["DestinationCityID"]
    demand_iters["Origin_Latitude"] = city_pair_data["Origin_Latitude"]
    demand_iters["Origin_Longitude"] = city_pair_data["Origin_Longitude"]
    demand_iters["Destination_Latitude"] = city_pair_data["Destination_Latitude"]
    demand_iters["Destination_Longitude"] = city_pair_data["Destination_Longitude"]
    demand_iters["base"] = city_pair_data["Total_Demand"]

    for iteration in range(maxiters):
        print(f"        Iteration {iteration+1} of fare optimisation")

        # allow airlines to adjust their fares
        airline_routes, city_pair_data = airline.optimise_fares(
            airlines,
            airline_routes,
            airline_fleets,
            city_pair_data,
            city_data,
            aircraft_data,
            max_fare,
            FuelCost_USDperGallon,
        )

        # write new columns to dataframes
        fare_iters[f"iter{iteration}"] = city_pair_data["Mean_Fare_USD"]
        demand_iters[f"iter{iteration}"] = city_pair_data["Total_Demand"]

        # write dataframes to files
        fare_iters.to_csv(os.path.join(save_folder_path, "fare_iters.csv"), index=False)
        demand_iters.to_csv(os.path.join(save_folder_path, "demand_iters.csv"), index=False)

        # check convergence
        if iteration > 0:
            diff = abs(demand_iters[f"iter{iteration}"] - demand_iters[f"iter{iteration-1}"])
            max_diff = diff.max()
            mean_diff = diff.mean()
            print(f"        Maximum demand shift: {max_diff}")
            print(f"        Mean demand shift: {mean_diff}")
            
            if max_diff < demand_tolerance:
                print(f"        Converged after {iteration} iterations")
                break


def run_simulation(
    airlines: pd.DataFrame,
    airline_fleets: list[pd.DataFrame],
    airline_routes: list[pd.DataFrame],
    city_data: pd.DataFrame,
    city_pair_data: pd.DataFrame,
    city_lookup: list[list[int]],
    aircraft_data: pd.DataFrame,
    demand_coefficients: dict[str, float],
    population_elasticity: float,
    population_data: pd.DataFrame,
    income_data: pd.DataFrame,
    fuel_data: pd.DataFrame,
    save_folder_path: str,
    base_year: int,
    end_year: int,
):
    max_fare = 50000.0

    FuelCost_USDperGallon = fuel_data.loc[
        fuel_data["Year"] == base_year, "Price_USD_per_Gallon"
    ].values[0]

    simulate_base_year(
        base_year,
        city_data,
        city_pair_data,
        airlines,
        airline_fleets,
        airline_routes,
        aircraft_data,
        FuelCost_USDperGallon,
        save_folder_path,
        max_fare,
    )

    # iterate over desired years
    for year in range(base_year + 1, end_year + 1):
        print(f"    Simulating year {year}...")

        # update data for the new year
        FuelCost_USDperGallon = fuel_data.loc[
            fuel_data["Year"] == year, "Price_USD_per_Gallon"
        ].values[0]
        airline_fleets = aircraft.annual_update(
            airlines,
            airline_fleets,
            aircraft_data,
            year,
        )
        city_data = city.annual_update(
            city_data,
            city_lookup,
            population_data,
            income_data,
            year,
        )
        city_pair_data = city.annual_update(
            city_pair_data,
            city_data,
            population_elasticity,
        )

        # allow airlines to adjust their fares and reassign aircraft to different routes
        airline_routes, city_pair_data = airline.optimise_fares(
            airlines,
            airline_routes,
            airline_fleets,
            city_pair_data,
            city_data,
            aircraft_data,
            max_fare,
            FuelCost_USDperGallon,
        )
        airline_routes, airline_fleets, city_pair_data, city_data = airline.reassign_ac_for_profit(
            airlines,
            airline_routes,
            airline_fleets,
            city_pair_data,
            city_data,
            city_lookup,
            aircraft_data,
            demand_coefficients,
            FuelCost_USDperGallon,
        )
