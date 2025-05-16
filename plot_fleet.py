import pandas as pd
import numpy as np
import os
import ast
import matplotlib.pyplot as plt


def plot_fleet_data(
    start_year: int,
    end_year: int,
    load_folder_path: str,
    output_folder_path: str,
) -> int:
    airlines = pd.read_csv(os.path.join(load_folder_path, f"airlines_{start_year}.csv"))
    airlines['n_Aircraft'] = airlines['n_Aircraft'].apply(ast.literal_eval)

    grounded_ac = np.zeros(end_year - start_year + 1)
    deployed_ac = {
        f"size_{i}": np.zeros(end_year - start_year + 1) for i in range(len(airlines["n_Aircraft"].iloc[0]))
    }

    year_range = list(range(start_year, end_year+1))

    for i in range(len(year_range)):
        airlines = pd.read_csv(os.path.join(load_folder_path, f"airlines_{year_range[i]}.csv"))
        airlines['n_Aircraft'] = airlines['n_Aircraft'].apply(ast.literal_eval)
        
        airlines['Grounded_acft'] = airlines['Grounded_acft'].apply(
            lambda x: [float(val.split('(')[1].split(')')[0]) for val in x.strip('[]').split(',') if 'float64' in val]
        )

        for idx, airline in airlines.iterrows():
            # get number of grounded aircraft
            grounded_ac[i] += len(airline["Grounded_acft"])

            # get number of deployed aircraft
            for size_idx, num in enumerate(airline["n_Aircraft"]):
                deployed_ac[f"size_{size_idx}"][i] += num

    plt.figure(figsize=(12, 6))
    plt.plot(year_range, grounded_ac, label="Grounded Aircraft (all sizes)", color="red", linestyle='dotted')
    for size_idx in range(len(airlines["n_Aircraft"].iloc[0])):
        plt.plot(year_range, deployed_ac[f"size_{size_idx}"], label=f"Size {size_idx}")

    plt.xlim(start_year, end_year)
    plt.ylim(bottom=0)
    plt.xticks(year_range)

    plt.xlabel("Year")
    plt.ylabel("Number of Aircraft Deployed")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # plt.show()

    output_filename = os.path.join(output_folder_path, "fleet_data.png")
    plt.savefig(output_filename, format='png', bbox_inches='tight')

    return 0



if __name__ == "__main__":
    start_year = 2015
    end_year = 2033
    load_folder_path = "output_BSL_15_05_PM"
    output_folder_path = "output_BSL_15_05_PM"
    plot_fleet_data(start_year, end_year, load_folder_path, output_folder_path)
