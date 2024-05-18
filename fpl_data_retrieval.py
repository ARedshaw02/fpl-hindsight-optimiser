import requests
import pandas as pd


def get_fpl_data():
    """
    Retrieves Fantasy Premier League data from the official API and returns it.

    Returns:
        dict: The data retrieved from the API.
    """
    # Define the URL for the API
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"

    # Send a GET request to the URL and retrieve the data
    response = requests.get(url)
    data = response.json()

    # Return the data
    return data
    
def get_current_gameweek(bootstrap_data):
    """
    Retrieve the current gameweek from the bootstrap data.

    Args:
        bootstrap_data (dict): The data retrieved from the API.

    Returns:
        int: The current gameweek.
    """
    # Loop through the events in the bootstrap data
    # and return the id of the event where is_current is true.
    # The next() function is used to stop after finding the first match.
    # The expression inside next() is a generator expression.
    # The generator expression iterates over the events in the bootstrap data
    # and yields the id of each event where is_current is true.
    # The next() function stops after the first yielded value.
    return next(event["id"] for event in bootstrap_data["events"] if event["is_current"])

def get_current_season(bootstrap_data):
    """
    Retrieve the current season from the bootstrap data.

    Args:
        bootstrap_data (dict): The data retrieved from the API.

    Returns:
        string: The current season.
    """
    current_year = int(bootstrap_data['events'][0]['deadline_time'][:4])

    return str(current_year) + '-' + str(current_year + 1)

def player_gameweek_data(start_gameweek, end_gameweek, bootstrap_data):
    """
    Retrieves player gameweek data from the official API for a given range of gameweeks.

    Args:
        start_gameweek (int): The starting gameweek to retrieve data for.
        end_gameweek (int): The ending gameweek to retrieve data for.
        bootstrap_data (dict): The data retrieved from the API.

    Returns:
        players_df (pd.DataFrame): The player gameweek data.
    """

    # Extract relevant information from bootstrap_data
    players_df = pd.DataFrame(bootstrap_data["elements"])
    players_df = players_df[["id", "web_name", "element_type", "now_cost", 
                              "cost_change_start", "team_code"]]

    # Calculate cost change from start of season
    cost_change_start = players_df["cost_change_start"]
    players_df["cost_change_start"] = players_df["now_cost"] - cost_change_start

    # Convert team_code to team short name
    teams = pd.DataFrame(bootstrap_data["teams"])[["code", "short_name"]]
    players_df = players_df.merge(teams, left_on="team_code", right_on="code", how="left")
    players_df.drop(columns=["team_code", "code"], inplace=True)

    # Rename cost_change_start to cost_start and add total_points
    players_df.rename(columns={"element_type" : "positions", "cost_change_start": "start_cost"}, inplace=True)
    players_df["total_points"] = 0

    # Map position IDs to position names
    position_map = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
    players_df["positions"] = players_df["positions"].map(position_map)

    # Retrieve gameweek data for each gameweek in the specified range
    for i in range(start_gameweek, end_gameweek + 1):
        # Construct API URL for the current gameweek
        url = f"https://fantasy.premierleague.com/api/event/{i}/live/"
        response = requests.get(url)
        data = response.json()

        # Create dictionaries to map player IDs to their points and minutes played
        points_dict = {element["id"]: element["stats"]["total_points"] for element in data["elements"]}
        minutes_dict = {element["id"]: element["stats"]["minutes"] for element in data["elements"]}

        # Add points and minutes to the players_df
        players_df[f"gw_{i}_points"] = players_df["id"].apply(lambda x: points_dict.get(x, 0))
        players_df[f"gw_{i}_minutes"] = players_df["id"].apply(lambda x: minutes_dict.get(x, 0))

        # Add points to total_points
        players_df["total_points"] += players_df[f"gw_{i}_points"]

    # Reset the index
    players_df = players_df.reset_index(drop=True)

    return players_df
