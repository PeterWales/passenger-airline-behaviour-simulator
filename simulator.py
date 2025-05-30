import pandas as pd
import airline as al
import aircraft
import city
import route
import demand
import os
import pickle
import datetime
import math
from constants import (
    FARE_CONVERGENCE_TOLERANCE,
    FUEL_ENERGY_MJ_KG,
    FUEL_GALLONS_PER_KG,
    PASSENGER_AC_FUEL_PROPORTION,
)


def simulate_base_year(
    year: int,
    city_data: pd.DataFrame,
    city_pair_data: pd.DataFrame,
    airlines: pd.DataFrame,
    airline_fleets: list[pd.DataFrame],
    airline_routes: list[pd.DataFrame],
    aircraft_data: pd.DataFrame,
    CJFCost_USDperGallon: float,
    save_folder_path: str,
    cache_folder_path: str,
    max_fare: float,
    iteration_limit: int,
    demand_coefficients: dict[str, float],
    saf_mandate_data,
    saf_pathway_data,
    run_parameters: dict,
    city_lookup: list[list[int]],
) -> tuple[list[pd.DataFrame], pd.DataFrame, float]:
    print(f"    Simulating base year ({year})...")
    print("    Time: ", datetime.datetime.now(), "\n")

    total_fuel_kg, city_pair_data, airline_routes = update_fuel_and_sales(
        airlines,
        airline_routes,
        airline_fleets,
        city_pair_data,
        city_data,
        aircraft_data,
    )

    if run_parameters["RunSAFMandate"] == "y" or run_parameters["RunSAFMandate"] == "Y":
        FuelCost_USDperGallon = fuel_price_with_saf(
            CJFCost_USDperGallon,
            saf_mandate_data,
            saf_pathway_data,
            year,
            total_fuel_kg,
        )
    else:
        FuelCost_USDperGallon = CJFCost_USDperGallon

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
        print(f"        Iteration {iteration+1} of initialisation...")

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

        # allow airlines to reassign aircraft to different routes (no addition or removal of aircraft)
        if run_parameters["LimitCapacity"] == "y" or run_parameters["LimitCapacity"] == "Y":
            unlimited_capacity = False
        else:
            unlimited_capacity = True
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
            unlimited_capacity,
            allow_lease_changes=False,
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

        # calculate fuel usage
        total_fuel_kg, city_pair_data, airline_routes = update_fuel_and_sales(
            airlines,
            airline_routes,
            airline_fleets,
            city_pair_data,
            city_data,
            aircraft_data,
            FuelCost_USDperGallon,
        )

        # recalculate fuel cost to account for change in fuel consumption
        if run_parameters["RunSAFMandate"] == "y" or run_parameters["RunSAFMandate"] == "Y":
            FuelCost_USDperGallon = fuel_price_with_saf(
                CJFCost_USDperGallon,
                saf_mandate_data,
                saf_pathway_data,
                year,
                total_fuel_kg,
            )
        else:
            FuelCost_USDperGallon = CJFCost_USDperGallon

        # write new columns to dataframes
        fare_iters[f"iter{iteration}"] = city_pair_data["Mean_Fare_USD"]
        demand_iters[f"iter{iteration}"] = city_pair_data["Total_Demand"]

        # write dataframes to files
        fare_iters.to_csv(os.path.join(save_folder_path, "fare_iters.csv"), index=False)
        demand_iters.to_csv(os.path.join(save_folder_path, "demand_iters.csv"), index=False)

        # check convergence
        if iteration > 0:
            diff = abs(fare_iters[f"iter{iteration}"] - fare_iters[f"iter{iteration-1}"])
            max_diff = diff.max()
            if max_diff > 0:
                mean_diff = diff[diff > 0].mean()
            else:
                mean_diff = 0.0
            print(f"        Maximum fare shift: {max_diff} USD")
            print(f"        Mean fare shift: {mean_diff} USD")
            
            if max_diff < FARE_CONVERGENCE_TOLERANCE:
                print(f"        Converged after {iteration} iterations")
                break
    
    fuel_data_out = pd.DataFrame(
        {
            "total_fuel_kg": [total_fuel_kg],
            "CJFCost_USDperGallon": [CJFCost_USDperGallon],
            "FuelCost_USDperGallon": [FuelCost_USDperGallon],
        }
    )
    fuel_data_out.to_csv(
        os.path.join(save_folder_path, f"fuel_data_{year}.csv"), index=False
    )
    
    # save pkl files
    with open(os.path.join(cache_folder_path, "airline_routes.pkl"), "wb") as f:
        pickle.dump(airline_routes, f)
    with open(os.path.join(cache_folder_path, "city_pair_data.pkl"), "wb") as f:
        pickle.dump(city_pair_data, f)
    with open(os.path.join(cache_folder_path, "total_fuel_kg.pkl"), "wb") as f:
        pickle.dump(total_fuel_kg, f)
    return airline_routes, city_pair_data, total_fuel_kg


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
    conv_fuel_data: pd.DataFrame,
    save_folder_path: str,
    cache_folder_path: str,
    airport_expansion_data: pd.DataFrame,
    saf_mandate_data: pd.DataFrame,
    saf_pathway_data: pd.DataFrame,
    run_parameters: dict,
):
    base_year = run_parameters["StartYear"]
    end_year = run_parameters["EndYear"]

    max_fare = run_parameters["MaxFare"]
    iteration_limit = run_parameters["BaseIterationLimit"]

    CJFCost_USDperGallon = conv_fuel_data.loc[
        conv_fuel_data["Year"] == base_year, "Price_USD_per_Gallon"
    ].values[0]
    
    if run_parameters["ContinueExistingSim"] == "n" or run_parameters["ContinueExistingSim"] == "N":
        start_year = base_year + 1

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
            airline_routes, city_pair_data, total_fuel_kg = simulate_base_year(
                base_year,
                city_data,
                city_pair_data,
                airlines,
                airline_fleets,
                airline_routes,
                aircraft_data,
                CJFCost_USDperGallon,
                save_folder_path,
                cache_folder_path,
                max_fare,
                iteration_limit,
                demand_coefficients,
                saf_mandate_data,
                saf_pathway_data,
                run_parameters,
                city_lookup,
            )
        else:
            with open(os.path.join(cache_folder_path, "total_fuel_kg.pkl"), "rb") as f:
                total_fuel_kg = pickle.load(f)

        # write dataframes to files
        city_data.to_csv(os.path.join(save_folder_path, f"city_data_{base_year}.csv"), index=True)
        city_pair_data.to_csv(os.path.join(save_folder_path, f"city_pair_data_{base_year}.csv"), index=False)
        airlines.to_csv(os.path.join(save_folder_path, f"airlines_{base_year}.csv"), index=False)
        # fuel_data_out saved inside simulate_base_year
    else:
        annual_cache_path = os.path.join(cache_folder_path, f"intermediate_{run_parameters['IDToContinue']}")
        with open(os.path.join(annual_cache_path, "year_completed.pkl"), "rb") as f:
            year_completed = pickle.load(f)
        if run_parameters["YearToContinue"] > year_completed + 1:
            start_year = year_completed + 1
        else:
            start_year = run_parameters["YearToContinue"]
        
        with open(os.path.join(annual_cache_path, f"total_fuel_kg_{start_year - 1}.pkl"), "rb") as f:
            total_fuel_kg = pickle.load(f)

    if run_parameters["LimitCapacity"] == "y" or run_parameters["LimitCapacity"] == "Y":
        unlimited_capacity = False
    else:
        unlimited_capacity = True

    # iterate over desired years
    for year in range(start_year, end_year + 1):
        print(f"\n    Simulating year {year}...")
        print("    Time: ", datetime.datetime.now(), "\n")

        # update data for the new year
        CJFCost_USDperGallon = conv_fuel_data.loc[
            conv_fuel_data["Year"] == year, "Price_USD_per_Gallon"
        ].values[0]

        if run_parameters["RunSAFMandate"] == "y" or run_parameters["RunSAFMandate"] == "Y":
            FuelCost_USDperGallon = fuel_price_with_saf(
                CJFCost_USDperGallon,
                saf_mandate_data,
                saf_pathway_data,
                year,
                total_fuel_kg,
            )
        else:
            FuelCost_USDperGallon = CJFCost_USDperGallon

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
            airport_expansion_data,
            year,
            run_parameters,
        )
        # city_pair_data must be updated after city_data because it needs new values from city_data
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

        # allow airlines to reassign aircraft to different routes, and add/retire aircraft
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
            unlimited_capacity,
            allow_lease_changes=True,
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

        # calculate fuel usage
        total_fuel_kg, city_pair_data, airline_routes = update_fuel_and_sales(
            airlines,
            airline_routes,
            airline_fleets,
            city_pair_data,
            city_data,
            aircraft_data,
            FuelCost_USDperGallon,
        )

        # write dataframes to files
        city_data.to_csv(os.path.join(save_folder_path, f"city_data_{year}.csv"), index=True)
        city_pair_data.to_csv(os.path.join(save_folder_path, f"city_pair_data_{year}.csv"), index=False)
        airlines.to_csv(os.path.join(save_folder_path, f"airlines_{year}.csv"), index=False)
        fuel_data_out = pd.DataFrame(
            {
                "total_fuel_kg": [total_fuel_kg],
                "CJFCost_USDperGallon": [CJFCost_USDperGallon],
                "FuelCost_USDperGallon": [FuelCost_USDperGallon],
            }
        )
        fuel_data_out.to_csv(
            os.path.join(save_folder_path, f"fuel_data_{year}.csv"), index=False
        )

        # save to pkl so simulation can be resumed from any year
        annual_cache_path = os.path.join(cache_folder_path, f"intermediate_{run_parameters['RunID']}")
        if not os.path.exists(annual_cache_path):
            os.makedirs(annual_cache_path)
        
        with open(os.path.join(annual_cache_path, f"airlines_{year}.pkl"), "wb") as f:
            pickle.dump(airlines, f)
        with open(os.path.join(annual_cache_path, f"airline_fleets_{year}.pkl"), "wb") as f:
            pickle.dump(airline_fleets, f)
        with open(os.path.join(annual_cache_path, f"airline_routes_{year}.pkl"), "wb") as f:
            pickle.dump(airline_routes, f)
        with open(os.path.join(annual_cache_path, f"city_data_{year}.pkl"), "wb") as f:
            pickle.dump(city_data, f)
        with open(os.path.join(annual_cache_path, f"city_pair_data_{year}.pkl"), "wb") as f:
            pickle.dump(city_pair_data, f)
        with open(os.path.join(annual_cache_path, f"total_fuel_kg_{year}.pkl"), "wb") as f:
            pickle.dump(total_fuel_kg, f)
        with open(os.path.join(annual_cache_path, "city_lookup.pkl"), "wb") as f:
            pickle.dump(city_lookup, f)
        with open(os.path.join(annual_cache_path, "country_data.pkl"), "wb") as f:
            pickle.dump(country_data, f)
        with open(os.path.join(annual_cache_path, "aircraft_data.pkl"), "wb") as f:
            pickle.dump(aircraft_data, f)
        with open(os.path.join(annual_cache_path, "year_completed.pkl"), "wb") as f:
            pickle.dump(year, f)


def limit_to_region(
    regions: list,
    airlines: pd.DataFrame,
    city_pair_data: pd.DataFrame,
    city_data: pd.DataFrame,
    country_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Adjust dataframes to limit the simulation to a certain region.

    Parameters
    ----------
    regions : list | None
        List of regions to include in the simulation. If None, all regions are included.
    airlines : pd.DataFrame
    city_pair_data : pd.DataFrame
    city_data : pd.DataFrame

    Returns
    -------
    airlines : pd.DataFrame
    city_pair_data : pd.DataFrame
    city_data : pd.DataFrame
    """
    if regions is not None:
        # drop airlines that are not registered in the region
        # airline_routes and airline_fleets shouldn't be altered since they are indexed by airline_id
        airlines = airlines[airlines["Region"].isin(regions)]
        
        # drop routes that are not entirely contained within the region
        to_drop = []
        for idx, route in city_pair_data.iterrows():
            origin = city_data.loc[route["OriginCityID"]]
            destination = city_data.loc[route["DestinationCityID"]]
            if origin["Region"] not in regions or destination["Region"] not in regions:
                to_drop.append(idx)
        city_pair_data.drop(to_drop, inplace=True)
        
        # drop cities that are not in the region
        city_data = city_data[city_data["Region"].isin(regions)]
        
        # calculate proportion of city movements attributed to routes that are not entirely contained within the region
        # these proportions are held constant throughout the simulation
        for idx, city in city_data.iterrows():
            if city["Movts_perHr"] == 0:
                city_data.at[idx, "Movts_Outside_Proportion"] = 0.0
            else:
                city_data.at[idx, "Movts_Outside_Proportion"] = city["Movts_Outside"] / city["Movts_perHr"]

    city_data = city_data.drop(columns=["Movts_Outside"])  # keep proportion only to avoid confusion

    return airlines, city_pair_data, city_data, country_data


def update_fuel_and_sales(
    airlines: pd.DataFrame,
    airline_routes: list[pd.DataFrame],
    airline_fleets: list[pd.DataFrame],
    city_pair_data: pd.DataFrame,
    city_data: pd.DataFrame,
    aircraft_data: pd.DataFrame,
    OldFuelCost_USDperGallon: float = -1.0,
) -> tuple[float, pd.DataFrame, list[pd.DataFrame]]:
    """
    Calculate the total fuel consumption for the simulation and update the "Fuel_Consumption_kg" column in the city_pair_data DataFrame.
    Also updates the "Tickets_Sold" column in the airline_routes and city_pair_data DataFrames.

    Parameters
    ----------
    airlines : pd.DataFrame
    airline_routes : list[pd.DataFrame]
    airline_fleets : list[pd.DataFrame]
    city_pair_data : pd.DataFrame
    aircraft_data : pd.DataFrame

    Returns
    -------
    total_fuel_kg : float
        Total fuel consumption in kg for the simulation.
    city_pair_data : pd.DataFrame
        Updated city_pair_data DataFrame with the "Fuel_Consumption_kg" column.
    """
    print("        Calculating fuel usage...")
    print("        Time: ", datetime.datetime.now(), "\n")
    
    # initialise fuel counting variables
    total_fuel_kg = 0.0
    city_pair_data["Fuel_Consumption_kg"] = 0.0

    city_pair_data["Tickets_Sold"] = 0

    # iterate over all airlines
    for airline_id, _ in airlines.iterrows():
        airline_routes[airline_id]["Tickets_Sold"] = 0
        airline_routes[airline_id]["Cost_per_Seat_Flight_USD"] = 0.0
        # iterate over all itineraries for the airline
        for al_route_idx, airline_route in airline_routes[airline_id].iterrows():
            city_pair_idx = city_pair_data.index[
                (city_pair_data["OriginCityID"] == airline_route["origin"])
                & (city_pair_data["DestinationCityID"] == airline_route["destination"])
            ].tolist()[0]
            city_pair = city_pair_data.loc[city_pair_idx]

            # calculate tickets sold (can't be more than the airline has scheduled or less than zero)
            annual_itin_demand = demand.update_itinerary_demand(city_pair, airline_route)
            tickets_sold = min([annual_itin_demand, airline_route["seat_flights_per_year"]])
            tickets_sold = max([0, tickets_sold])

            airline_routes[airline_id].at[al_route_idx, "Tickets_Sold"] = tickets_sold
            city_pair_data.at[city_pair_idx, "Tickets_Sold"] += tickets_sold

            # calculate total seats for all aircraft on route
            planes = airline_route["aircraft_ids"]
            fleet_df = airline_fleets[airline_id]
            total_seats = 0
            for acft_id in planes:
                acft = fleet_df.loc[fleet_df["AircraftID"] == acft_id].iloc[0]
                aircraft_type = aircraft_data.loc[acft["SizeClass"]]
                total_seats += aircraft_type["Seats"]
            
            if OldFuelCost_USDperGallon > 0:
                annual_itin_cost = aircraft.calc_flight_cost(
                    airline_route,
                    fleet_df,
                    aircraft_data,
                    city_pair,
                    city_data.loc[airline_route["origin"]],
                    city_data.loc[airline_route["destination"]],
                    annual_itin_demand,
                    OldFuelCost_USDperGallon,
                )
                airline_routes[airline_id].at[al_route_idx, "Cost_per_Seat_Flight_USD"] = annual_itin_cost / airline_route["seat_flights_per_year"]

            # iterate over all aircraft assigned to itinerary
            for acft_id in planes:
                acft = fleet_df.loc[fleet_df["AircraftID"] == acft_id].iloc[0]
                aircraft_type = aircraft_data.loc[acft["SizeClass"]]

                itin_demand_share = float(aircraft_type["Seats"] * annual_itin_demand) / total_seats
                pax_perflt_share = math.floor(itin_demand_share / acft["Flights_perYear"])
                pax = min([pax_perflt_share, aircraft_type["Seats"]])
                pax = max([pax, 0])
        
                fuel_per_flt = aircraft.calc_fuel_consumption(
                    aircraft_type,
                    acft,
                    city_pair,
                    pax,
                )

                fuel_per_year = fuel_per_flt * acft["Flights_perYear"]
                total_fuel_kg += fuel_per_year
                city_pair_data.at[city_pair_idx, "Fuel_Consumption_kg"] += fuel_per_year
    
    return total_fuel_kg, city_pair_data, airline_routes


def fuel_price_with_saf(
    CJFCost_USDperGallon: float,
    saf_mandate_data: pd.DataFrame,
    saf_pathway_data: pd.DataFrame,
    year: int,
    total_fuel_kg: float,
):
    """
    Calculate the fuel price including mandated SAF proportion.
    After satisfying the synthetic sub-mandate, the cheapest SAF pathways are used first.
    Assume cargo aircraft and private jets are subject to the same SAF mandate as passenger aircraft, but military aircraft are not.

    Parameters
    ----------
    CJFCost_USDperGallon : float
        Conventional jet fuel price in USD per gallon.
    saf_mandate_data : pd.DataFrame
        DataFrame containing year-by-year SAF mandate and synthetic sub-mandate.
    saf_pathway_data : pd.DataFrame
        DataFrame containing data on individual SAF production pathways.
    year : int
        Year of the simulation.
    total_fuel_kg : float
        Total fuel consumption in kg in the previous year of simulation.
    
    Returns
    -------
    FuelCost_USDperGallon : float
        Average fuel price in USD per gallon including CJF and SAF.
    """
    if year < saf_mandate_data["Year"].min():
        # no SAF mandate in this year
        return CJFCost_USDperGallon
    
    saf_mandate_data = saf_mandate_data.sort_values(by="Year")

    if year > saf_mandate_data["Year"].max():
        # use the last SAF mandate value for all future years
        SAF_Mandate = saf_mandate_data["SAF_Mandate"].values[-1]
        Synth_Mandate = saf_mandate_data["Synthetic_Sub_Mandate"].values[-1]

    SAF_Mandate = saf_mandate_data.loc[saf_mandate_data["Year"] == year, "SAF_Mandate"].values[0]
    Synth_Mandate = saf_mandate_data.loc[saf_mandate_data["Year"] == year, "Synthetic_Sub_Mandate"].values[0]

    if SAF_Mandate == 0:
        # no SAF mandate in this year
        return CJFCost_USDperGallon
    
    # calculate price in the current year of each SAF pathway
    saf_pathway_data["Cost_USDperkg_current"] = 0.0
    saf_pathway_data["Max_Output_Mt"] = 0.0
    for idx, pathway in saf_pathway_data.iterrows():
        # linearly interpolate cost
        cost_2030 = pathway["Cost_USDperMJ_2030"]
        cost_2050 = pathway["Cost_USDperMJ_2050"]
        cost_USDperMJ = cost_2030 + (cost_2050 - cost_2030) * (year - 2030) / (2050 - 2030)
        saf_pathway_data.at[idx, "Cost_USDperkg_current"] = cost_USDperMJ * FUEL_ENERGY_MJ_KG

        if pathway["Pathway"] == "PtL":
            saf_pathway_data.at[idx, "Max_Output_Mt"] = -1.0  # PtL pathway is not limited by feedstock availability
        else:
            # linearly interpolate feedstock availability between Mt_Feedstock_2030 and Mt_Feedstock_2050
            feedstock_2030 = pathway["Mt_Feedstock_2030"]
            feedstock_2050 = pathway["Mt_Feedstock_2050"]
            feedstock_current = feedstock_2030 + (feedstock_2050 - feedstock_2030) * (year - 2030) / (2050 - 2030)
            saf_pathway_data.at[idx, "Max_Output_Mt"] = feedstock_current * pathway["SAF_Fraction"] * pathway["Fuel_Yield"]
    
    # sort pathways by cost
    saf_pathway_data = saf_pathway_data.sort_values(by="Cost_USDperkg_current")

    # determine average price using cheapest pathways first after meeting synthetic sub-mandate
    saf_to_assign = SAF_Mandate * total_fuel_kg / PASSENGER_AC_FUEL_PROPORTION  # scale fuel usage to account for cargo and private jets
    saf_total_cost_USD = 0.0
    if Synth_Mandate > 0:
        kg_from_pathway = Synth_Mandate * total_fuel_kg
        saf_to_assign -= kg_from_pathway
        saf_total_cost_USD += kg_from_pathway * saf_pathway_data.loc[
            saf_pathway_data["Pathway"] == "PtL", "Cost_USDperkg_current"
        ].values[0]

    for idx, pathway in saf_pathway_data.iterrows():
        if saf_to_assign <= 0:
            break
        if pathway["Pathway"] == "PtL":
            # PtL pathway is not limited by feedstock availability
            kg_from_pathway = saf_to_assign
        else:
            kg_from_pathway = min([pathway["Max_Output_Mt"] * 1e9, saf_to_assign])  # 1 Mt = 1 million tonnes = 1e9 kg
        saf_to_assign -= kg_from_pathway
        saf_total_cost_USD += kg_from_pathway * pathway["Cost_USDperkg_current"]

    # adjust SAF total cost back to passenger aircraft only
    saf_passenger_cost_USD = saf_total_cost_USD * PASSENGER_AC_FUEL_PROPORTION

    # include cost of conventional jet fuel
    cjf_kg = (1-SAF_Mandate) * total_fuel_kg
    cjf_total_cost_USD = cjf_kg * FUEL_GALLONS_PER_KG * CJFCost_USDperGallon

    FuelCost_USDperkg = (saf_passenger_cost_USD + cjf_total_cost_USD) / total_fuel_kg
    FuelCost_USDperGallon = FuelCost_USDperkg / FUEL_GALLONS_PER_KG

    return FuelCost_USDperGallon
