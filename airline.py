from dataclasses import dataclass
import pandas as pd
import numpy as np


def initialise_airlines(
    fleet_data: pd.DataFrame,
    country_data: pd.DataFrame,
    run_parameters: pd.DataFrame,
) -> list:
    """
    Generate a DataFrame of airline data from contents of FleetData and CountryData files
    Within each region, the number of aircraft of each type assigned to each country is proportional to the country's GDP
    Within each country, the aircraft of each type are assigned to the airlines as evenly as possible

    Parameters
    ----------
    fleet_data : pd.DataFrame
    country_data : pd.DataFrame
    run_parameters : pd.DataFrame

    Returns
    -------
    airlines : pd.DataFrame
    """
    census_regions = {
        "Americas": [10, 11, 12],
        "Europe": [13],
        "MiddleEast": [14],
        "Africa": [15],
        "AsiaPacific": [16],
    }

    country_data["GDP"] = (
        country_data["PPP_GDP_Cap_Year2015USD_2015"]
        * country_data["Population_2015"]
    )
    country_data["Region_GDP_Proportion"] = 0.0  # initialise with zeros

    airlines = pd.DataFrame()
    airline_idx = 0
    n_aircraft_types = len(fleet_data)

    airline_id = []
    region_id = []
    country_name = []
    country_id = []
    aircraft_lists = []
    aircraft = []
    airline_routes = []

    for region, codes in census_regions.items():
        # calculate GDP proportions
        countries = country_data["Region"].isin(codes)
        if not any(countries):
            continue
        region_gdp = country_data.loc[countries, "GDP"].sum()
        country_data.loc[countries, "Region_GDP_Proportion"] = (
            country_data.loc[countries, "GDP"] / region_gdp
        )

        # sort countries in region by GDP
        sorted_region = sorted(
            country_data.loc[countries].iterrows(),
            key=lambda x: (
                x[1]["GDP"],
                x[1]["Population_2015"],
            ),  # Use population as tiebreaker
            reverse=True,
        )

        # assign aircraft to countries
        region_aircraft = fleet_data[f"Census{region}"].to_list()
        unallocated = region_aircraft.copy()
        for (country_idx, country,) in sorted_region:
            country_aircraft = [0] * n_aircraft_types
            # iterate through aircraft types and account for rounding issues
            for i in range(n_aircraft_types):
                allocated = int(
                    round(region_aircraft[i] * country["Region_GDP_Proportion"])
                )
                if allocated == 0 and unallocated[i] > 0:
                    allocated = 1
                country_aircraft[i] = min(allocated, unallocated[i])
                unallocated[i] -= country_aircraft[i]

            # assign aircraft to airlines within country
            for country_airline_idx in range(run_parameters["AirlinesPerCountry"]):
                n_aircraft = [0] * n_aircraft_types
                for aircraft_type in range(n_aircraft_types):
                    base = (
                        country_aircraft[aircraft_type]
                        // run_parameters["AirlinesPerCountry"]
                    )
                    remainder = (
                        country_aircraft[aircraft_type]
                        % run_parameters["AirlinesPerCountry"]
                    )
                    # add one extra aircraft to each airline until the remainder is used up
                    n_aircraft[aircraft_type] = base + (
                        1 if country_airline_idx < remainder else 0
                    )

                # create airline, unless it has no aircraft
                if sum(n_aircraft) > 0:
                    airline_id.append(airline_idx)
                    region_id.append(country["Region"])
                    country_name.append(country["Country"])
                    country_id.append(country["Number"])
                    aircraft_lists.append(n_aircraft)
                    aircraft.append([])
                    airline_routes.append(
                        pd.DataFrame(
                            columns=[
                                "origin",
                                "destination",
                                "fare",
                                "aircraft_ids",
                                "seat_flights_per_year",
                            ]
                        )
                    )
                    airline_idx += 1

    airlines["Airline_ID"] = airline_id
    airlines["Region"] = region_id
    airlines["Country"] = country_name
    airlines["CountryID"] = country_id
    airlines["n_Aircraft"] = aircraft_lists
    airlines["Aircraft"] = aircraft
    airlines["Airline_Routes"] = airline_routes

    return airlines


def initialise_fleet_assignment(
    airlines: list,
    routes: list,
    cities: list,
    aircraft_data: pd.DataFrame,
    city_lookup: list,
    randomGen: np.random.Generator,
    year: int,
) -> None:
    """
    Assign aircraft to routes based on base demand, aircraft range and runway required

    Parameters
    ----------
    routes : list
        list of lists of Route instances
    cities : list
        list of City instances
    aircraft_data : pd.DataFrame
        dataframe containing aircraft data
    city_lookup : list
        list of lists of city IDs by country code
    randomGen : np.random.Generator
        random number generator
    year : int
        calendar year
    
    Updates self.aircraft, self.airline_routes, and Route instances in-place
    """
    # TODO: enable EU airlines to operate routes between all EU countries
    # TODO: move aircraft creation and following lines into a function to avoid code duplication

    min_load_factor = 0.8  # minimum load factor for an aircraft to be assigned to a route

    # calculate total base RPKs for all routes the airline can operate (assume airlines can only run routes to/from their home country)
    possible_RPKs = 0.0
    distances = []
    n_cities = len(routes)
    for origin_id in city_lookup[self.country_num]:
        for destination_id in range(n_cities):
            if routes[origin_id][destination_id] is not None:
                # outbound flight
                route_RPKs = (
                    routes[origin_id][destination_id].base_demand
                    * routes[origin_id][destination_id].distance
                )
                # inbound flight
                route_RPKs += (
                    routes[destination_id][origin_id].base_demand
                    * routes[destination_id][origin_id].distance
                )
                possible_RPKs += route_RPKs
                # append a tuple(origin_id, destination_id, distance, RPKs) to distances list
                distances.append((origin_id, destination_id, routes[origin_id][destination_id].distance, route_RPKs))

    # calculate total airline seat capacity
    total_seats = 0
    for aircraft_type, n_aircraft_type in enumerate(self.n_aircraft):
        total_seats += n_aircraft_type * aircraft_data.at[aircraft_type, "Seats"]

    # assign aircraft seat capacity by base demand starting with the largest aircraft on the longest routes
    distances.sort(key=lambda x: x[2], reverse=True)
    aircraft_avail = self.n_aircraft.copy()
    aircraft_id = -1
    for origin_id, destination_id, distance, route_RPKs in distances:
        # stop if no aircraft available
        if sum(aircraft_avail) == 0:
            break
        
        seats = total_seats * (route_RPKs / possible_RPKs)

        for _, aircraft in aircraft_data.iterrows():
            if seats <= 0:
                break
            aircraft_size = aircraft["AircraftID"]
            if aircraft_avail[aircraft_size] > 0:
                # check aircraft has enough range
                if distance < aircraft["TypicalRange_m"]:  # must be kept seperate due to else statement
                    # check origin and destination runways are long enough
                    if (
                        (routes[origin_id][destination_id].origin.longest_runway_m > aircraft["TakeoffDist_m"])
                        and (routes[origin_id][destination_id].origin.longest_runway_m > aircraft["LandingDist_m"])
                        and (routes[origin_id][destination_id].destination.longest_runway_m > aircraft["TakeoffDist_m"])
                        and (routes[origin_id][destination_id].destination.longest_runway_m > aircraft["LandingDist_m"])
                    ):
                        while(
                            (seats / aircraft["Seats"] > min_load_factor)
                            and (aircraft_avail[aircraft_size] > 0)
                        ):
                            seats -= aircraft["Seats"]  # can go negative
                            aircraft_avail[aircraft_size] -= 1
                            # create aircraft instance and assign to route, randomise age
                            aircraft_id += 1
                            self.aircraft.append(
                                Aircraft(
                                    aircraft_id=aircraft_id,
                                    aircraft_size=aircraft_size,
                                    seats=aircraft["Seats"],
                                    lease_new_USDpermonth=aircraft["LeaseRateNew_USDPerMonth"],
                                    lease_annualmultiplier=aircraft["LeaseRateAnnualMultiplier"],
                                    age=randomGen.randint(0, aircraft["RetirementAge_years"]-1),
                                    operation_USDperhr=aircraft["OpCost_USDPerHr"],
                                    pilot_USDperhr=aircraft["PilotCost_USDPerPilotPerHr"],
                                    crew_USDperhr=aircraft["CrewCost_USDPerCrewPerHr"],
                                    maintenance_daysperyear=aircraft["MaintenanceDays_PerYear"],
                                    turnaround_hrs=aircraft["Turnaround_hrs"],
                                    op_empty_kg=aircraft["OEM_kg"],
                                    max_fuel_kg=aircraft["MaxFuel_kg"],
                                    max_payload_kg=aircraft["MaxPayload_kg"],
                                    max_takeoff_kg=aircraft["MTOM_kg"],
                                    cruise_ms=aircraft["CruiseV_ms"],
                                    ceiling_ft=aircraft["Ceiling_ft"],
                                    breguet_gradient=aircraft["Breguet_gradient"],
                                    breguet_intercept=aircraft["Breguet_intercept"],
                                    retirement_age=aircraft["RetirementAge_years"],
                                    calendar_year=year,
                                    route_origin=origin_id,
                                    route_destination=destination_id,
                                    fuel_stop=None,
                                )
                            )
                            self.aircraft[-1].update_flights_per_year(routes)

                            # update Route in-place
                            routes[origin_id][destination_id].seat_flights_per_year += (
                                self.aircraft[-1].flights_per_year * self.aircraft[-1].seats
                            )

                            # update airline-specific route dataframe
                            not_route_exists = self.airline_routes[
                                (self.airline_routes["origin"] == origin_id)
                                & (self.airline_routes["destination"] == destination_id)
                            ].empty
                            if not_route_exists:
                                self.airline_routes = self.airline_routes.append(
                                    {
                                        "origin": origin_id,
                                        "destination": destination_id,
                                        "fare": routes[origin_id][destination_id].base_mean_fare,
                                        "aircraft_ids": [aircraft_id],
                                        "seat_flights_per_year": self.aircraft[-1].flights_per_year * self.aircraft[-1].seats,
                                    },
                                    ignore_index=True,
                                )
                            else:
                                self.airline_routes.loc[
                                    (self.airline_routes["origin"] == origin_id)
                                    & (self.airline_routes["destination"] == destination_id),
                                    "aircraft_ids"
                                ].append(aircraft_id)
                                self.airline_routes.loc[
                                    (self.airline_routes["origin"] == origin_id)
                                    & (self.airline_routes["destination"] == destination_id),
                                    "seat_flights_per_year"
                                ] += self.aircraft[-1].flights_per_year * self.aircraft[-1].seats

                elif distance < 2*aircraft["TypicalRange_m"]:
                    # aircraft doesn't have enough range - try with a fuel stop
                    fuel_stop = Route.find_fuel_stop(
                        routes,
                        cities,
                        origin_id,
                        destination_id,
                        aircraft["TypicalRange_m"],
                        min(
                            aircraft["TakeoffDist_m"],
                            aircraft["LandingDist_m"]
                        )
                    )
                    if fuel_stop is not None:  # None if route not possible wih a stop for that aircraft
                        while(
                            (seats / aircraft["Seats"] > min_load_factor)
                            and (aircraft_avail[aircraft_size] > 0)
                        ):
                            seats -= aircraft["Seats"]  # can go negative
                            aircraft_avail[aircraft_size] -= 1
                            # create aircraft instance and assign to route, randomise age
                            aircraft_id += 1
                            self.aircraft.append(
                                Aircraft(
                                    aircraft_id=aircraft_id,
                                    aircraft_size=aircraft_size,
                                    seats=aircraft["Seats"],
                                    lease_new_USDpermonth=aircraft["LeaseRateNew_USDPerMonth"],
                                    lease_annualmultiplier=aircraft["LeaseRateAnnualMultiplier"],
                                    age=randomGen.randint(0, aircraft["RetirementAge_years"]-1),
                                    operation_USDperhr=aircraft["OpCost_USDPerHr"],
                                    pilot_USDperhr=aircraft["PilotCost_USDPerPilotPerHr"],
                                    crew_USDperhr=aircraft["CrewCost_USDPerCrewPerHr"],
                                    maintenance_daysperyear=aircraft["MaintenanceDays_PerYear"],
                                    turnaround_hrs=aircraft["Turnaround_hrs"],
                                    op_empty_kg=aircraft["OEM_kg"],
                                    max_fuel_kg=aircraft["MaxFuel_kg"],
                                    max_payload_kg=aircraft["MaxPayload_kg"],
                                    max_takeoff_kg=aircraft["MTOM_kg"],
                                    cruise_ms=aircraft["CruiseV_ms"],
                                    ceiling_ft=aircraft["Ceiling_ft"],
                                    breguet_gradient=aircraft["Breguet_gradient"],
                                    breguet_intercept=aircraft["Breguet_intercept"],
                                    retirement_age=aircraft["RetirementAge_years"],
                                    calendar_year=year,
                                    route_origin=origin_id,
                                    route_destination=destination_id,
                                    fuel_stop=fuel_stop,
                                )
                            )
                            self.aircraft[-1].update_flights_per_year(routes)

                            # update Route in-place
                            routes[origin_id][destination_id].seat_flights_per_year += (
                                self.aircraft[-1].flights_per_year * self.aircraft[-1].seats
                            )

                            # update airline-specific route dataframe
                            not_route_exists = self.airline_routes[
                                (self.airline_routes["origin"] == origin_id)
                                & (self.airline_routes["destination"] == destination_id)
                            ].empty
                            if not_route_exists:
                                self.airline_routes = self.airline_routes.append(
                                    {
                                        "origin": origin_id,
                                        "destination": destination_id,
                                        "fare": routes[origin_id][destination_id].base_mean_fare,
                                        "aircraft_ids": [aircraft_id],
                                        "seat_flights_per_year": self.aircraft[-1].flights_per_year * self.aircraft[-1].seats,
                                    },
                                    ignore_index=True,
                                )
                            else:
                                self.airline_routes.loc[
                                    (self.airline_routes["origin"] == origin_id)
                                    & (self.airline_routes["destination"] == destination_id),
                                    "aircraft_ids"
                                ].append(aircraft_id)
                                self.airline_routes.loc[
                                    (self.airline_routes["origin"] == origin_id)
                                    & (self.airline_routes["destination"] == destination_id),
                                    "seat_flights_per_year"
                                ] += self.aircraft[-1].flights_per_year * self.aircraft[-1].seats

                else:
                    break
