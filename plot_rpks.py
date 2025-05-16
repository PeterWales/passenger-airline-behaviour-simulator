import pandas as pd
import os
import matplotlib.pyplot as plt


def plot_rpks(
    start_year: int,
    end_year: int,
    max_flight_distance_km: float,
    load_folder_path: str,
    output_folder_path: str,
) -> int:
    year_range = list(range(start_year, end_year+1))

    RPKs = []
    ASKs = []
    total_demand = []
    total_seat_flights = []
    ASK_weighted_price = []
    for i in range(len(year_range)):
        year_ASKs = 0.0
        year_RPKs = 0.0
        year_demand = 0.0
        year_seat_flights = 0.0
        year_price_sum = 0.0
        city_pair_data = pd.read_csv(os.path.join(load_folder_path, f"city_pair_data_{year_range[i]}.csv"))
        for _, row in city_pair_data.iterrows():
            route_distance_km = row["Great_Circle_Distance_m"] / 1000
            if route_distance_km <= max_flight_distance_km:
                route_seat_flights = row["Seat_Flights_perYear"]
                if route_seat_flights > 0:
                    year_seat_flights += route_seat_flights
                    route_ASKs = route_seat_flights * route_distance_km
                    year_ASKs += route_ASKs
                    route_demand = max(row["Total_Demand"], 0.0)
                    year_demand += route_demand
                    route_RPKs = min(route_demand * route_distance_km, route_ASKs)  # RPKs can never be more than ASKs
                    year_RPKs += route_RPKs
                    year_price_sum += row["Mean_Fare_USD"] * route_ASKs
        RPKs.append(year_RPKs / 1e9)  # convert to billions
        ASKs.append(year_ASKs / 1e9)
        total_demand.append(year_demand / 1e6)  # convert to millions
        total_seat_flights.append(year_seat_flights / 1e6)  # convert to millions
        ASK_weighted_price.append(year_price_sum / year_ASKs if year_ASKs > 0 else 0.0)

    plt.figure(figsize=(12, 6))
    plt.plot(year_range, ASK_weighted_price)

    plt.xlim(start_year, end_year)
    plt.ylim(bottom=0)
    plt.xticks(year_range)

    plt.xlabel("Year")
    plt.ylabel("ASK-weighted average Ticket Price (USD)")

    # plt.show()

    output_filename = os.path.join(output_folder_path, "mean_ticket_price.png")
    plt.savefig(output_filename, format='png', bbox_inches='tight')


    plt.figure(figsize=(12, 6))
    
    # Create the first axis for RPKs and ASKs
    ax1 = plt.gca()
    ax1.plot(year_range, RPKs, label="RPKs", color='blue')
    ax1.plot(year_range, ASKs, label="ASKs", color='green')
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Total RPKs and ASKs (Billions)")
    
    # Create the secondary axis for demand
    ax2 = ax1.twinx()
    ax2.plot(year_range, total_demand, label="Total Ticket Demand", color='orange', linestyle='--')
    ax2.plot(year_range, total_seat_flights, label="Total Seat Flights", color='red', linestyle='--')
    ax2.set_ylabel("Total Demand and Seat Flights (Millions)", color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # Set common x-axis properties
    ax1.set_xlim(start_year, end_year)
    ax1.set_ylim(bottom=0)
    ax2.set_ylim(bottom=0)
    ax1.set_xticks(year_range)
    
    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, bbox_to_anchor=(1.15, 1), loc='upper left')
    
    plt.tight_layout()

    # plt.show()

    output_filename = os.path.join(output_folder_path, "rpk_data.png")
    plt.savefig(output_filename, format='png', bbox_inches='tight')

    return 0



if __name__ == "__main__":
    start_year = 2015
    end_year = 2033
    max_flight_distance_km = 6000.0
    # max_flight_distance_km = 40000.0
    load_folder_path = "output_BSL_15_05_PM"
    output_folder_path = "output_BSL_15_05_PM"
    plot_rpks(start_year, end_year, max_flight_distance_km, load_folder_path, output_folder_path)
