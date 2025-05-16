import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


def plot_city_data(
    start_year: int,
    end_year: int,
    sample_year: int,
    n_cities: int,
    load_folder_path: str,
    output_folder_path: str,
) -> int:
    cities = pd.read_csv(os.path.join(load_folder_path, f"city_data_{sample_year}.csv"))
    
    cities = cities.sort_values(by="Movts_perHr", ascending=False)
    cities = cities.head(n_cities)
    cityIDs_to_plot = cities["CityID"].tolist()

    city_movements = {
        f"{city_name}": np.zeros(end_year - start_year + 1) for city_name in cities["CityName"]
    }

    year_range = list(range(start_year, end_year+1))

    for i in range(len(year_range)):
        cities = pd.read_csv(os.path.join(load_folder_path, f"city_data_{year_range[i]}.csv"))
        cities = cities[cities["CityID"].isin(cityIDs_to_plot)]

        for _, city in cities.iterrows():
            city_movements[city["CityName"]][i] = city["Movts_perHr"]

    plt.figure(figsize=(12, 6))

    if n_cities < 20:
        colors = plt.cm.tab10(np.linspace(0, 1, n_cities))
    else:
        colors = plt.cm.tab20(np.linspace(0, 1, n_cities))
    line_style_set = ['-', '--', ':', '-.']
    line_styles = [line_style_set[i % len(line_style_set)] for i in range(n_cities)]
    style_cycler = plt.cycler(color=colors) + plt.cycler(linestyle=line_styles)
    plt.gca().set_prop_cycle(style_cycler)

    for city_name, mvmts in city_movements.items():
        plt.plot(year_range, mvmts, label=city_name)

    plt.xlim(start_year, end_year)
    plt.ylim(bottom=0)
    plt.xticks(year_range)

    plt.xlabel("Year")
    plt.ylabel("Mean Movements per Hour (sum of all airports in city)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # plt.show()

    output_filename = os.path.join(output_folder_path, "city_data.png")
    plt.savefig(output_filename, format='png', bbox_inches='tight')

    return 0



if __name__ == "__main__":
    start_year = 2015
    end_year = 2033
    sample_year = 2020  # for choosing which cities to plot
    n_cities = 15
    load_folder_path = "output_BSL_15_05_PM"
    output_folder_path = "output_BSL_15_05_PM"
    plot_city_data(start_year, end_year, sample_year, n_cities, load_folder_path, output_folder_path)
