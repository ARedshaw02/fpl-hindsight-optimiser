from fpl_data_retrieval import get_fpl_data, get_current_gameweek, player_gameweek_data
import sasoptpy as so
import pandas as pd

def all_player_data(start_gameweek, end_gameweek):
    """
    Retrieves player gameweek data from the official API for a given range of gameweeks.

    Args:
        start_gameweek (int): The starting gameweek to retrieve data for.
        end_gameweek (int): The ending gameweek to retrieve data for.

    Returns:
        pd.DataFrame: The player gameweek data.
    """
    # Retrieve FPL data from the official API
    data = get_fpl_data()

    # Get the current gameweek
    current_gameweek = get_current_gameweek(data)

    # If the end_gameweek is greater than the current gameweek, set it to the current gameweek
    if end_gameweek > current_gameweek:
        end_gameweek = current_gameweek

    # Retrieve player gameweek data for the specified range of gameweeks
    return player_gameweek_data(start_gameweek, end_gameweek, data)

def basic_set_and_forget(player_gameweek_df, start_gameweek, end_gameweek):
    pass

if __name__ == "__main__":
    min_gameweek = 1
    max_gameweek = 38
    player_gameweek_df = all_player_data(min_gameweek, max_gameweek)

    print(player_gameweek_df)
