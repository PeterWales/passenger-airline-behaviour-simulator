import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


def plot_routes(
    start_year: int,
    end_year: int,
    load_folder_path: str,
    output_folder_path: str,
) -> int:
    routes = pd.read_csv(os.path.join(load_folder_path, f"city_pair_data_{start_year}.csv"))
    cities = pd.read_csv(os.path.join(load_folder_path, f"city_data_{start_year}.csv"))

    routes = routes[(routes["OriginCityID"] == 3) | (routes["DestinationCityID"] == 3)]

    routes = routes.sort_values(by="Seat_Flights_perYear", ascending=False)
    routes = routes.head(15)
    route_indices_to_plot = routes.index.tolist()

    route_names = []
    for _, route in routes.iterrows():
        origin_name = cities.loc[cities["CityID"] == route["OriginCityID"], "CityName"].values[0]
        destination_name = cities.loc[cities["CityID"] == route["DestinationCityID"], "CityName"].values[0]
        route_names.append(f"{origin_name} - {destination_name}")

    route_prices = {
        f"{route_name}": np.zeros(end_year - start_year + 1) for route_name in route_names
    }

    route_seat_flights = {
        f"{route_name}": np.zeros(end_year - start_year + 1) for route_name in route_names
    }

    year_range = list(range(start_year, end_year+1))

    for i in range(len(year_range)):
        routes = pd.read_csv(os.path.join(load_folder_path, f"city_pair_data_{year_range[i]}.csv"))
        routes = routes[routes.index.isin(route_indices_to_plot)]

        j = 0
        for idx in route_indices_to_plot:
            route = routes.loc[idx]
            route_prices[route_names[j]][i] = route["Mean_Fare_USD"]
            route_seat_flights[route_names[j]][i] = route["Seat_Flights_perYear"]
            j += 1
    
    pricefig = plt.figure(figsize=(12, 6))
    for route_name, prices in route_prices.items():
        plt.plot(year_range, prices, label=route_name)

    plt.xlim(start_year, end_year)
    plt.ylim(bottom=0)

    plt.xlabel("Year")
    plt.ylabel("Mean Fare (USD)")
    plt.legend()

    output_filename = os.path.join(output_folder_path, "route_prices.png")
    plt.savefig(output_filename, format='png', bbox_inches='tight')
    plt.close(pricefig)

    seatfig = plt.figure(figsize=(12, 6))
    for route_name, seat_flights in route_seat_flights.items():
        plt.plot(year_range, seat_flights, label=route_name)

    plt.xlim(start_year, end_year)
    plt.ylim(bottom=0)

    plt.xlabel("Year")
    plt.ylabel("Seat Flights per Year")
    plt.legend()

    output_filename = os.path.join(output_folder_path, "route_seat_flights.png")
    plt.savefig(output_filename, format='png', bbox_inches='tight')
    plt.close(seatfig)

    return 0


if __name__ == "__main__":
    start_year = 2015
    end_year = 2036
    load_folder_path = "output_Ref_EU"
    output_folder_path = "output_Ref_EU"
    plot_routes(start_year, end_year, load_folder_path, output_folder_path)
