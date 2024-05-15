import requests
import json
import pandas as pd
import numpy as np

def get_fpl_data():
    """
    Retrieves Fantasy Premier League data from the official API and returns it.

    Returns:
        dict: The data retrieved from the API.
    """
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    response = requests.get(url)
    data = response.json()
    return data
    
def get_current_gameweek(bootstrap_data):
    """
    Retrieves the current gameweek from the bootstrap data.

    Args:
        bootstrap_data (dict): The data retrieved from the API.

    Returns:
        int: The current gameweek.
    """
    # Find event where is_current is true and return the id
    return next(event["id"] for event in bootstrap_data["events"] if event["is_current"])

def player_gameweek_data(start_gameweek, end_gameweek, bootstrap_data):
    """
    Retrieves player gameweek data from the official API for a given player and gameweek.

    Args:
        start_gameweek (int): The starting gameweek to retrieve data for.
        end_gameweek (int): The ending gameweek to retrieve data for.
        bootstrap_data (dict): The data retrieved from the API.

    Returns:
        players_df (pd.DataFrame): The player gameweek data.
    """

    players_df = pd.DataFrame(bootstrap_data["elements"])
    # Keep only the relevant columns

    players_df = players_df[["id", "web_name", "element_type", "now_cost", "cost_change_start"]]
    cost_change_start = players_df["cost_change_start"]
    players_df["cost_change_start"] = players_df["now_cost"] - cost_change_start

    # Rename cost_change_start to cost_start
    players_df.rename(columns={"element_type" : "positions" ,"cost_change_start": "start_cost"}, inplace=True)
    players_df["total_points"] = 0

    # Map position IDs to position names
    position_map = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
    players_df["positions"] = players_df["positions"].map(position_map)

    for i in range(start_gameweek, end_gameweek + 1):
        url = f"https://fantasy.premierleague.com/api/event/{i}/live/"
        response = requests.get(url)
        data = response.json()

        # Create a dictionary to map player IDs to their points
        points_dict = {element["id"]: element["stats"]["total_points"] for element in data["elements"]}

        # Use the dictionary to add the players' gameweek points to a column titled gw_{i}
        players_df[f"gw_{i}"] = players_df["id"].apply(lambda x: points_dict.get(x, 0))

        # Add the players' gameweek points to their total points
        players_df["total_points"] += players_df[f"gw_{i}"]

    return players_df

if __name__ == "__main__":
    data = get_fpl_data()
    start_gameweek = 1
    end_gameweek = 38 
    current_gameweek = get_current_gameweek(data)
    if end_gameweek > current_gameweek:
        end_gameweek = current_gameweek

    player_gameweek_df = player_gameweek_data(start_gameweek, current_gameweek, data)
    
    player_data = {}
    for index, row in player_gameweek_df.iterrows():
        player_data[row["id"]] = {
            "name": row["web_name"],
            "position": row["positions"],
            "start_cost": row["start_cost"],
            "total_points": row["total_points"]
        }
