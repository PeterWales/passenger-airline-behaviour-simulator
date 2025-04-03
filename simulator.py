import pandas as pd
import airline as al
import aircraft
import city
import route
import os
import pickle


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
    cache_folder_path: str,
    max_fare: float,
    iteration_limit: int,
    demand_coefficients: dict[str, float],
) -> tuple[list[pd.DataFrame], pd.DataFrame]:
    print(f"    Simulating base year ({year})...")

    # parameters for finding Nash equilibrium
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

    for iteration in range(iteration_limit):
        print(f"        Iteration {iteration+1} of fare optimisation")

        # allow airlines to adjust their fares
        airline_routes, city_pair_data = al.optimise_fares(
            airlines,
            airline_routes,
            airline_fleets,
            city_pair_data,
            city_data,
            aircraft_data,
            max_fare,
            FuelCost_USDperGallon,
            demand_coefficients,
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
    
    # save pkl files
    with open(os.path.join(cache_folder_path, "airline_routes.pkl"), "wb") as f:
                pickle.dump(airline_routes, f)
    with open(os.path.join(cache_folder_path, "city_pair_data.pkl"), "wb") as f:
        pickle.dump(city_pair_data, f)
    return airline_routes, city_pair_data


def run_simulation(
    airlines: pd.DataFrame,
    airline_fleets: list[pd.DataFrame],
    airline_routes: list[pd.DataFrame],
    city_data: pd.DataFrame,
    city_pair_data: pd.DataFrame,
    city_lookup: list[list[int]],
    country_data,
    aircraft_data: pd.DataFrame,
    demand_coefficients: dict[str, float],
    population_elasticity: float,
    population_data: pd.DataFrame,
    income_data: pd.DataFrame,
    fuel_data: pd.DataFrame,
    save_folder_path: str,
    cache_folder_path: str,
    run_parameters: dict,
):
    base_year = run_parameters["StartYear"]
    end_year = run_parameters["EndYear"]

    max_fare = run_parameters["MaxFare"]
    iteration_limit = run_parameters["BaseIterationLimit"]

    FuelCost_USDperGallon = fuel_data.loc[
        fuel_data["Year"] == base_year, "Price_USD_per_Gallon"
    ].values[0]

    for al_idx, _ in airlines.iterrows():
        airline_routes[al_idx]["itin_time_hrs"] = 0.0

    # update airline itinerary times
    airline_routes = al.update_itinerary_times(
        airlines,
        airline_routes,
        city_data,
        city_pair_data,
        aircraft_data,
        airline_fleets,
    )

    if (
        run_parameters["RerunFareInit"] == "y"
        or run_parameters["RerunFareInit"] == "Y"
    ):
        airline_routes, city_pair_data = simulate_base_year(
            base_year,
            city_data,
            city_pair_data,
            airlines,
            airline_fleets,
            airline_routes,
            aircraft_data,
            FuelCost_USDperGallon,
            save_folder_path,
            cache_folder_path,
            max_fare,
            iteration_limit,
            demand_coefficients,
        )

    # write dataframes to files
    city_data.to_csv(os.path.join(save_folder_path, f"city_data_{base_year}.csv"), index=True)
    city_pair_data.to_csv(os.path.join(save_folder_path, f"city_pair_data_{base_year}.csv"), index=False)
    airlines.to_csv(os.path.join(save_folder_path, f"airlines_{base_year}.csv"), index=False)

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
            country_data,
            city_data,
            city_lookup,
            population_data,
            income_data,
            year,
        )
        city_pair_data = route.annual_update(
            city_pair_data,
            city_data,
            population_elasticity,
        )

        # allow airlines to adjust their fares and reassign aircraft to different routes
        airline_routes, city_pair_data = al.optimise_fares(
            airlines,
            airline_routes,
            airline_fleets,
            city_pair_data,
            city_data,
            aircraft_data,
            max_fare,
            FuelCost_USDperGallon,
            demand_coefficients,
        )
        airline_routes, airline_fleets, city_pair_data, city_data = al.reassign_ac_for_profit(
            airlines,
            airline_routes,
            airline_fleets,
            city_pair_data,
            city_data,
            city_lookup,
            aircraft_data,
            demand_coefficients,
            FuelCost_USDperGallon,
            year,
        )

        # update airline itinerary times
        airline_routes = al.update_itinerary_times(
            airlines,
            airline_routes,
            city_data,
            city_pair_data,
            aircraft_data,
            airline_fleets,
        )

        # write dataframes to files
        city_data.to_csv(os.path.join(save_folder_path, f"city_data_{year}.csv"), index=True)
        city_pair_data.to_csv(os.path.join(save_folder_path, f"city_pair_data_{year}.csv"), index=False)
        airlines.to_csv(os.path.join(save_folder_path, f"airlines_{year}.csv"), index=False)
