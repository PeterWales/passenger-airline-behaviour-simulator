import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


def plot_routes(
    start_year: int,
    end_year: int,
    sample_year: int,
    n_routes: int,
    load_folder_path: str,
    output_folder_path: str,
) -> int:
    routes = pd.read_csv(os.path.join(load_folder_path, f"city_pair_data_{sample_year}.csv"))
    cities = pd.read_csv(os.path.join(load_folder_path, f"city_data_{sample_year}.csv"))

    routes = routes.sort_values(by="Seat_Flights_perYear", ascending=False)
    # routes = routes.head(n_routes)
    # route_indices_to_plot = routes.index.tolist()
    outbound_indices = []
    return_indices = []
    i = 0
    for idx, route in routes.iterrows():
        if idx not in outbound_indices and idx not in return_indices:
            outbound_indices.append(idx)
            return_indices.append(routes[(routes["OriginCityID"] == route["DestinationCityID"]) & (routes["DestinationCityID"] == route["OriginCityID"])].index[0])
            i += 1
        if i >= n_routes:
            break

    out_route_names = []
    return_route_names = []
    all_route_names = []
    for idx in outbound_indices:
        route = routes.loc[idx]
        origin_name = cities.loc[cities["CityID"] == route["OriginCityID"], "CityName"].values[0]
        destination_name = cities.loc[cities["CityID"] == route["DestinationCityID"], "CityName"].values[0]
        out_route_names.append(f"{origin_name} - {destination_name}")
        return_route_names.append(f"{destination_name} - {origin_name}")
        all_route_names.append(f"{origin_name} - {destination_name}")
        all_route_names.append(f"{destination_name} - {origin_name}")

    route_prices = {
        f"{route_name}": np.zeros(end_year - start_year + 1) for route_name in all_route_names
    }

    route_seat_flights = {
        f"{route_name}": np.zeros(end_year - start_year + 1) for route_name in out_route_names
    }

    year_range = list(range(start_year, end_year+1))

    for i in range(len(year_range)):
        routes = pd.read_csv(os.path.join(load_folder_path, f"city_pair_data_{year_range[i]}.csv"))

        j = 0
        for k in range(len(outbound_indices)):
            out_idx = outbound_indices[k]
            return_idx = return_indices[k]
            out_route = routes.loc[out_idx]
            return_route = routes.loc[return_idx]
            route_prices[out_route_names[j]][i] = out_route["Mean_Fare_USD"]
            route_prices[return_route_names[j]][i] = return_route["Mean_Fare_USD"]
            route_seat_flights[out_route_names[j]][i] = out_route["Seat_Flights_perYear"]
            j += 1
    
    colors_temp = plt.cm.tab20(np.linspace(0, 1, n_routes))
    colors = np.repeat(colors_temp, 2, axis=0)
    line_style_set = ['-', '--']
    line_styles = [line_style_set[i % len(line_style_set)] for i in range(n_routes * 2)]
    style_cycler = plt.cycler(color=colors) + plt.cycler(linestyle=line_styles)
    
    pricefig = plt.figure(figsize=(12, 6))
    plt.gca().set_prop_cycle(style_cycler)
    for route_name, prices in route_prices.items():
        plt.plot(year_range, prices, label=route_name)

    plt.xlim(start_year, end_year)
    plt.ylim(bottom=0)
    plt.xticks(year_range)

    plt.xlabel("Year")
    plt.ylabel("Mean Fare (USD)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    output_filename = os.path.join(output_folder_path, "route_prices.png")
    plt.savefig(output_filename, format='png', bbox_inches='tight')
    plt.close(pricefig)


    seatfig = plt.figure(figsize=(12, 6))

    if n_routes < 20:
        colors = plt.cm.tab10(np.linspace(0, 1, n_routes))
    else:
        colors = plt.cm.tab20(np.linspace(0, 1, n_routes))
    line_style_set = ['-', '--', ':', '-.']
    line_styles = [line_style_set[i % len(line_style_set)] for i in range(n_routes)]
    style_cycler = plt.cycler(color=colors) + plt.cycler(linestyle=line_styles)
    plt.gca().set_prop_cycle(style_cycler)

    plt.gca().set_prop_cycle(style_cycler)
    for route_name, seat_flights in route_seat_flights.items():
        plt.plot(year_range, seat_flights, label=route_name)

    plt.xlim(start_year, end_year)
    plt.ylim(bottom=0)
    plt.xticks(year_range)

    plt.xlabel("Year")
    plt.ylabel("Seat Flights per Year")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    output_filename = os.path.join(output_folder_path, "route_seat_flights.png")
    plt.savefig(output_filename, format='png', bbox_inches='tight')
    plt.close(seatfig)

    return 0


if __name__ == "__main__":
    start_year = 2015
    end_year = 2033
    sample_year = 2020  # for choosing which routes to plot
    n_routes = 10
    load_folder_path = "output_BSL_15_05_PM"
    output_folder_path = "output_BSL_15_05_PM"
    plot_routes(start_year, end_year, sample_year, n_routes, load_folder_path, output_folder_path)
