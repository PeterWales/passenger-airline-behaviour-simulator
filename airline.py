from dataclasses import dataclass
import pandas as pd


@dataclass
class Airline:
    """
    Class for keeping track of airline data

    Attributes
    ----------
    airline_id : int
    region : int
    country : str
    country_num : int
    n_aircraft : list
        Number of aircraft of each type assigned to the airline
    
    Methods
    -------
    initialise_airlines(fleet_data, country_data, run_parameters)
        Generate list of instances of Airline dataclass from contents of FleetData and CountryData files
    """

    airline_id: int
    region: int
    country: str
    country_num: int
    n_aircraft: list

    @staticmethod
    def initialise_airlines(
        fleet_data: pd.DataFrame,
        country_data: pd.DataFrame,
        run_parameters: pd.DataFrame,
    ) -> list:
        """
        Generate list of instances of Airline dataclass from contents of FleetData and CountryData files
        Within each region, the number of aircraft of each type assigned to each country is proportional to the country's GDP
        Within each country, the aircraft of each type are assigned to the airlines as evenly as possible

        Parameters
        ----------
        fleet_data : pd.DataFrame
        country_data : pd.DataFrame
        run_parameters : pd.DataFrame

        Returns
        -------
        airlines : list of instances of Airline dataclass
        """
        census_regions = {
            "Americas": [10, 11, 12],
            "Europe": [13],
            "MiddleEast": [14],
            "Africa": [15],
            "AsiaPacific": [16],
        }

        country_data["GDP"] = (
            country_data["PPP_GDP_Cap_Year2015USD_2015"]
            * country_data["Population_2015"]
        )
        country_data["Region_GDP_Proportion"] = 0.0  # initialise with zeros

        airlines = []
        airline_idx = 0
        n_aircraft_types = len(fleet_data)

        for region, codes in census_regions.items():
            # calculate GDP proportions
            countries = country_data["Region"].isin(codes)
            if not any(countries):
                continue
            region_gdp = country_data.loc[countries, "GDP"].sum()
            country_data.loc[countries, "Region_GDP_Proportion"] = (
                country_data.loc[countries, "GDP"] / region_gdp
            )

            # sort countries in region by GDP
            sorted_region = sorted(
                country_data.loc[countries].iterrows(),
                key=lambda x: (
                    x[1]["GDP"],
                    x[1]["Population_2015"],
                ),  # Use population as tiebreaker
                reverse=True,
            )

            # assign aircraft to countries
            region_aircraft = fleet_data[f"Census{region}"].to_list()
            unallocated = region_aircraft.copy()
            for (
                country_idx,
                country,
            ) in sorted_region:  # no need for iterrows due to sorting method
                country_aircraft = [0] * n_aircraft_types
                # iterate through aircraft types and account for rounding issues
                for i in range(n_aircraft_types):
                    allocated = int(
                        round(region_aircraft[i] * country["Region_GDP_Proportion"])
                    )
                    if allocated == 0 and unallocated[i] > 0:
                        allocated = 1
                    country_aircraft[i] = min(allocated, unallocated[i])
                    unallocated[i] -= country_aircraft[i]

                # assign aircraft to airlines within country
                for country_airline_idx in range(run_parameters["AirlinesPerCountry"]):
                    n_aircraft = [0] * n_aircraft_types
                    for aircraft_type in range(n_aircraft_types):
                        base = (
                            country_aircraft[aircraft_type]
                            // run_parameters["AirlinesPerCountry"]
                        )
                        remainder = (
                            country_aircraft[aircraft_type]
                            % run_parameters["AirlinesPerCountry"]
                        )
                        # add one extra aircraft to each airline until the remainder is used up
                        n_aircraft[aircraft_type] = base + (
                            1 if country_airline_idx < remainder else 0
                        )

                    # create airline, unless it has no aircraft
                    if sum(n_aircraft) > 0:
                        airlines.append(
                            Airline(
                                airline_id=airline_idx,
                                region=country["Region"],
                                country=country["Country"],
                                country_num=country["Number"],
                                n_aircraft=n_aircraft,
                            )
                        )

                        airline_idx += 1
        return airlines
