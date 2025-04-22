import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


def plot_city_data(
    start_year: int,
    end_year: int,
    load_folder_path: str,
    output_folder_path: str,
) -> int:
    cities = pd.read_csv(os.path.join(load_folder_path, f"city_data_{start_year}.csv"))

    city_movements = {
        f"{city_name}": np.zeros(end_year - start_year + 1) for city_name in cities["CityName"]
    }

    year_range = list(range(start_year, end_year+1))

    for i in range(len(year_range)):
        cities = pd.read_csv(os.path.join(load_folder_path, f"city_data_{year_range[i]}.csv"))

        for _, city in cities.iterrows():
            city_movements[city["CityName"]][i] = city["Movts_perHr"]

    plt.figure(figsize=(12, 6))
    for city_name, mvmts in city_movements.items():
        plt.plot(year_range, mvmts, label=city_name)

    plt.xlim(start_year, end_year)
    plt.ylim(bottom=0)

    plt.xlabel("Year")
    plt.ylabel("Mean Movements per Hour (sum of all airports in city)")
    plt.legend()

    # plt.show()

    output_filename = os.path.join(output_folder_path, "city_data.png")
    plt.savefig(output_filename, format='png', bbox_inches='tight')

    return 0



if __name__ == "__main__":
    start_year = 2015
    end_year = 2025
    load_folder_path = "output_Ref"
    output_folder_path = "output_Ref"
    plot_city_data(start_year, end_year, load_folder_path, output_folder_path)
