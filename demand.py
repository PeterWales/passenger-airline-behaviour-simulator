import pandas as pd
import math


def update_od_demand(
    city_pair: pd.Series,
) -> float:
    """
    Update the total route demand based on fare and annual static factors.

    Parameters
    ----------
    city_pair : pd.Series
        A row from the city_pair_data DataFrame
    
    Returns
    -------
    demand
    """
    # TODO: add input for national taxes
    fare_factor = 1 + (
        ((city_pair["Mean_Fare_USD"] - city_pair["Fare_Est"]) / city_pair["Fare_Est"])
        * city_pair['Price_Elasticity_Route']
    )
    # tax_factor = 1 + ((delta_tax / self.mean_fare) * self.price_elasticities["national"])
    demand = math.floor(city_pair["BaseYearODDemandPax_Est"] * fare_factor * city_pair["Static_Demand_Factor"])
    return demand


def update_itinerary_demand(
    city_pair: pd.Series,
    airline_route: pd.Series,
) -> float:
    """
    Update the annual demand for a particular itinerary based on the options for an O-D pair.

    Parameters
    ----------
    route : pd.Series
        Series of O-D pair data (row from the city_pair_data DataFrame)
    airline_route : pd.Series
        Series of airline route itinerary data (row from the airline_routes DataFrame)

    Returns
    -------
    demand : float
        demand in pax per year for the itinerary
    """
    market_share = airline_route["exp_utility"] / city_pair["Exp_Utility_Sum"]
    demand = math.floor(city_pair["Total_Demand"] * market_share)
    return demand


def calc_exp_utility(
    demand_coefficients: dict[str, float],
    fare: float,
    flight_time_hrs: float,
    flights_per_year: int,
    fuel_stop: int,
):
    """
    Calculate the e^utility of a flight segment.

    Parameters
    ----------
    demand_coefficients : dict
        Dictionary of demand coefficients
    fare : float
        Fare for the segment
    flight_time_hrs : float
        Flight time in hours
    flights_per_year : int
        Number of flights per year
    fuel_stop : int
        ID of city where aircraft must stop to refuel, or -1 if non-stop
    
    Returns
    -------
    exp_util : float
    """
    if flights_per_year == 0:
        exp_utility = 0
    elif flights_per_year < 0:
        exp_utility = 0
        print(f"NEGATIVE FLIGHTS PER YEAR: {flights_per_year}")
    else:
        if fuel_stop == -1:
            segment_term = demand_coefficients["mu"]
        else:
            segment_term = 2*demand_coefficients["mu"]

        utility = (
            demand_coefficients["theta"]*fare
            + demand_coefficients["k"]*flight_time_hrs
            + demand_coefficients["lambda"]*math.log(flights_per_year)
            + segment_term
        )

        exp_utility = math.exp(utility)
    return exp_utility
