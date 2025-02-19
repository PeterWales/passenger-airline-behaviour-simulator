import pandas as pd
import math


def update_od_demand(
    route: pd.Series,
) -> float:
    """
    Update the total route demand based on fare and annual static factors.

    Parameters
    ----------
    route : pd.Series
        A row from the city_pair_data DataFrame
    
    Returns
    -------
    demand
    """
    # TODO: add input for national taxes
    fare_factor = 1 + (
        ((route["Mean_Fare_USD"] - route["Fare_Est"]) / route["Fare_Est"])
        * route['Price_Elasticity_Route']
    )
    # tax_factor = 1 + ((delta_tax / self.mean_fare) * self.price_elasticities["national"])
    demand = route["BaseYearODDemandPax_Est"] * fare_factor * route["Static_Demand_Factor"]
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
    market_share = airline_route["exp_utility"] / city_pair["exp_utility_sum"]
    demand = city_pair["Total_Demand"] * market_share
    return demand


def calc_exp_utility(
    demand_coefficients: dict[str, float],
    fare: float,
    flight_time_hrs: float,
    flights_per_year: int,
    fuel_stop: None | int,
):
    if fuel_stop is None:
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
