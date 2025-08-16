import os
import pandas as pd
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap, PowerNorm
import numpy as np


def map_plot(
    plot_year: int,
    load_folder_path: str,
    output_folder_path: str,
    long_range: list | None,
    lat_range: list | None
):
    city_pair_data = pd.read_csv(os.path.join(load_folder_path, f"city_pair_data_{plot_year}.csv"))

    # define colours
    background = (1.0, 1.0, 1.0, 1.0)
    coastline = (10/255.0, 10/255.0, 10/255.0, 0.8)
    color = (204/255.0, 0.0, 153/255.0, 0.6)

    # color_points = [
    #     (1.0, 1.0, 1.0, 0.0),
    #     (204/255.0, 0, 153/255.0, 0.6),
    #     (255/255.0, 204/255.0, 230/255.0, 1.0)
    # ]
    # n_colors = city_pair_data["Seat_Flights_perYear"].max()
    # cmap = LinearSegmentedColormap.from_list(
    #     'cmap_flights',
    #     color_points,
    #     N=n_colors
    # )
    
    # norm = PowerNorm(
    #     0.3,
    #     city_pair_data['Seat_Flights_perYear'].min(),
    #     city_pair_data['Seat_Flights_perYear'].max()
    # )

    # linewidth = 0.5

    max_seat_flights = city_pair_data['Seat_Flights_perYear'].max()
    min_linewidth = 0.2
    linewidth_range = 0.5

    # create a map with coastlines
    plt.figure(figsize=(27, 20))
    m = Basemap(projection='mill', lon_0=0)
    m.drawcoastlines(color=coastline, linewidth=1.0)
    m.fillcontinents(color=background, lake_color=background)
    m.drawmapboundary(fill_color=background)


    # plot great circle routes
    for i, city_pair in city_pair_data.iterrows():
        if city_pair['Seat_Flights_perYear'] > 0:
            # color = cmap(norm(int(city_pair['Seat_Flights_perYear'])))
            linewidth = ((city_pair['Seat_Flights_perYear'] / max_seat_flights) * linewidth_range) + min_linewidth

            gc = m.gcpoints(city_pair["Origin_Longitude"], 
                        city_pair["Origin_Latitude"],
                        city_pair["Destination_Longitude"], 
                        city_pair["Destination_Latitude"],
                        100)  # number of points
            
            # Draw the line using the calculated points
            line, = m.plot(gc[0], gc[1], 
                        linewidth=linewidth,
                        color=color)

            # fix for routes that go off the edge of the map (plotted path > 30,000km)
            path = line.get_path()
            cut_point, = np.where(np.abs(np.diff(path.vertices[:, 0])) > 30000e3)
            if len(cut_point) > 0:
                cut_point = cut_point[0]
                vertices = np.concatenate([path.vertices[:cut_point, :],
                                        [[np.nan, np.nan]],
                                        path.vertices[cut_point+1:, :]])
                path.codes = None  # treat vertices as a series of line segments
                path.vertices = vertices
        
    # longitude/latitude limits
    if (
        long_range is not None
        and lat_range is not None
    ):
        x_min, y_min = m(long_range[0], lat_range[0])  # Convert lon/lat to map projection
        x_max, y_max = m(long_range[1], lat_range[1])
        plt.xlim([x_min, x_max])
        plt.ylim([y_min, y_max])

    # plt.show()

    # save the map
    output_filename = os.path.join(output_folder_path, f"map_{plot_year}.png")
    plt.savefig(output_filename, format='png', bbox_inches='tight')

    return 0



if __name__ == "__main__":
    plot_year = 2016
    load_folder_path = "output_BSL_15_05_AM"
    output_folder_path = "output_BSL_15_05_AM"

    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    # Europe
    long_range = [-20, 40]
    lat_range = [30, 75]

    map_plot(plot_year, load_folder_path, output_folder_path, long_range, lat_range)
