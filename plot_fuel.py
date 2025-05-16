import pandas as pd
import os
import matplotlib.pyplot as plt


def plot_fuel_data(
    start_year: int,
    end_year: int,
    load_folder_path: str,
    output_folder_path: str,
) -> int:
    year_range = list(range(start_year, end_year+1))

    fuel_usage_Mt = []
    for i in range(len(year_range)):
        fuel_data = pd.read_csv(os.path.join(load_folder_path, f"fuel_data_{year_range[i]}.csv"))
        fuel_usage_Mt.append(fuel_data["total_fuel_kg"].iloc[0] / 1e9)

    plt.figure(figsize=(12, 6))
    plt.plot(year_range, fuel_usage_Mt)

    plt.xlim(start_year, end_year)
    plt.ylim(bottom=0)
    plt.xticks(year_range)

    plt.xlabel("Year")
    plt.ylabel("Total Fuel Usage (Mt)")

    # plt.show()

    output_filename = os.path.join(output_folder_path, "fuel_data.png")
    plt.savefig(output_filename, format='png', bbox_inches='tight')

    return 0



if __name__ == "__main__":
    start_year = 2015
    end_year = 2027
    load_folder_path = "output_BSL_15_05_AM"
    output_folder_path = "output_BSL_15_05_AM"
    plot_fuel_data(start_year, end_year, load_folder_path, output_folder_path)
