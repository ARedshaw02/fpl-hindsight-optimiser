from fpl_data_retrieval import get_fpl_data, get_current_gameweek, player_gameweek_data
import pulp as plp
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

def basic_set_and_forget(player_gameweek_df, bench_multiplier, budget=1000):
    """
    Solves the basic 'set and forget problem' using PuLP. The set and forget problem considers a team
    that is chosen from gameweek 1, with a budget of 100m, and no further changes are made.
    This basic modelling does not explicitly consider substitutions when players do not feature, 
    or vice captain swapping when the captain does not feature. For this reason, it will not be fully optimal, 
    but the use of the bench multiplier, and later comparisons of different values for this, should enable
    optimisation to a relatively high level.

    Args:
        player_gameweek_df (pd.DataFrame): The player gameweek data.
        bench_multiplier (float): The multiplier for bench players (also applied to the vice captain).
        budget (float): The budget for the team (million value divided by 0.1m).

    Returns:
        Tuple[List[LpVariable], List[LpVariable], List[LpVariable], List[LpVariable]]: The decision variables for lineup, bench, captaincy, and vice_captaincy.
    """
    df = player_gameweek_df[["id", "total_points", "short_name", "positions", "start_cost"]]   
    
    player_ids = df['id'].tolist()
    player_count = len(player_ids)

    # Set up the problem
    model = plp.LpProblem("basic-set-forget", plp.LpMaximize)

    # Define the decision variables
    lineup = [
        plp.LpVariable(f"lineup_{i}", lowBound=0, upBound=1, cat="Integer")
        for i in player_ids
    ]
    captaincy = [
        plp.LpVariable(f"captaincy_{i}", lowBound=0, upBound=1, cat="Integer")
        for i in player_ids
    ]
    vice_captaincy = [
        plp.LpVariable(f"vice_captaincy_{i}", lowBound=0, upBound=1, cat="Integer")
        for i in player_ids
    ]
    bench = [
        plp.LpVariable(f"bench_{i}", lowBound=0, upBound=1, cat="Integer")
        for i in player_ids
    ]

    # Set the objective function
    model += sum((lineup[i] + captaincy[i] + (bench_multiplier * vice_captaincy[i]) + (bench_multiplier * bench[i])) * df["total_points"][i] for i in range(player_count))

    # Set the constraints
    model += sum((lineup[i] + bench[i]) * df["start_cost"][i] for i in range(player_count)) <= budget

     # GK constraints
    model += sum(lineup[i] for i in range(player_count) if df['positions'][i] == 'GK') == 1
    model += sum(lineup[i] + bench[i] for i in range(player_count) if df['positions'][i] == 'GK') == 2

    # DEF constraints
    model += sum(lineup[i] for i in range(player_count) if df['positions'][i] == 'DEF') >= 3
    model += sum(lineup[i] + bench[i] for i in range(player_count) if df['positions'][i] == 'DEF') == 5

    # MID constraints
    model += sum(lineup[i] for i in range(player_count) if df['positions'][i] == 'MID') >= 2
    model += sum(lineup[i] + bench[i] for i in range(player_count) if df['positions'][i] == 'MID') == 5

    # FWD constraints
    model += sum(lineup[i] for i in range(player_count) if df['positions'][i] == 'FWD') >= 1
    model += sum(lineup[i] + bench[i] for i in range(player_count) if df['positions'][i] == 'FWD') == 3

    # Team constraints
    model += sum(lineup) == 11
    model += sum(bench) == 4
    model += sum(lineup) + sum(bench) == 15
    model += sum(captaincy) == 1
    model += sum(vice_captaincy) == 1
    for teams in df['short_name'].unique():
        model += sum(lineup[i] + bench[i] for i in range(player_count) if df['short_name'][i] == teams) <= 3
    
    for i in range(player_count):
        model += (lineup[i] + bench[i]) <= 1
        model += (lineup[i] - captaincy[i]) >= 0
        model += (lineup[i] - vice_captaincy[i]) >=0
        model += (captaincy[i] + vice_captaincy[i]) <= 1

    plp.LpSolverDefault.msg = 0
    model.solve()

    return lineup, bench, captaincy, vice_captaincy

def retrieve_base_id(decision_var):
    """
    Retrieve the base id from the decision variable.

    Args:
        decision_var (str): The decision variable.

    Returns:
        int: The base id.
    """
    # Remove all non-digit characters from the decision variable.
    decision_var = ''.join(i for i in decision_var if i.isdigit())

    # Convert the string to an integer and return.
    return int(decision_var)

def retrieve_refactored_model_output(lineup, bench, captaincy, vice_captaincy):
    """
    Retrieve the output of the basic set and forget model after solving.
    
    Args:
        lineup (list): The lineup decision variables.
        bench (list): The bench decision variables.
        captaincy (list): The captaincy decision variables.
        vice_captaincy (list): The vice captaincy decision variables.
    
    Returns:
        dict: A dictionary containing the lineup, bench, captaincy, and vice captaincy player ids.
    """
    # Define the variables and mapping
    variables = [lineup, bench, captaincy, vice_captaincy]

    # Define the mapping from variable index to variable name
    mapping_dict = {
        0 : 'lineup',
        1 : 'bench',
        2 : 'captaincy',
        3 : 'vice_captaincy'
    }

    # Define the dictionary to store the output
    return_dict = {
        'lineup': [],
        'bench': [],
        'captaincy': [],
        'vice_captaincy': []
    }

    # Loop over the variables and retrieve the player ids
    for i, variable in enumerate(variables):
        for j in range(len(variable)):
            # Check if the variable is selected
            if variable[j].value() == 1:
                # Retrieve the player id and add it to the return dictionary
                return_dict[mapping_dict[i]].append(retrieve_base_id(variable[j].name))
    
    # Return the output dictionary
    return return_dict

def retrieve_model_gameweek_history(model_output, player_gameweek_df):
    """
    Retrieve the player gameweek history from the model output.

    Args:
        model_output (dict): The output of the model.
        player_gameweek_df (pd.DataFrame): The player gameweek data.

    Returns:
        pd.DataFrame: The player gameweek history.
    """
    # Add binary column values to indicate if the player is in the lineup, bench, captaincy, or vice captaincy
    player_gameweek_df['in_lineup'] = player_gameweek_df['id'].isin(model_output['lineup'])
    player_gameweek_df['on_bench'] = player_gameweek_df['id'].isin(model_output['bench'])
    player_gameweek_df['is_captain'] = player_gameweek_df['id'].isin(model_output['captaincy'])
    player_gameweek_df['is_vice_captain'] = player_gameweek_df['id'].isin(model_output['vice_captaincy'])

    # Retrieve the player ids from the model output
    player_ids = model_output['lineup'] + model_output['bench'] + model_output['captaincy'] + model_output['vice_captaincy']

    # Filter the player gameweek data based on the player ids
    return player_gameweek_df[player_gameweek_df['id'].isin(player_ids)]

def simulate_model_team(model_players_df):
    # Check for the current gameweek from the outset, to prevent repeated api calls.
    current_gw = get_current_gameweek(get_fpl_data())

    def get_bench_player(df, bench_ids, pos='*'):
        """
        Retrieves the first player from the bench list that satisfies the given position.

        Args:
            df (pandas.DataFrame): The player data.
            bench_ids (list): The list of player ids on the bench.
            pos (str): The position to look for ('DEF', 'MID', 'FWD', or '*' for any position).

        Returns:
            int or None: The player id if found, None otherwise.
        """
        # If the bench list is empty, return None
        if len(bench_ids) == 0:
            return None

        # If the position is '*', return the first player in the bench list
        if pos == '*':
            return bench_ids[0]

        # If the position is 'DEF', return the first player in the bench list that is a defender
        if pos == 'DEF':
            for id in bench_ids:
                if id in df[(df['positions'] == 'DEF')]['id'].tolist():
                    return id

        # If the position is 'MID', return the first player in the bench list that is a midfielder
        if pos == 'MID':
            for id in bench_ids:
                if id in df[(df['positions'] == 'MID')]['id'].tolist():
                    return id

        # If the position is 'FWD', return the first player in the bench list that is a forward
        if pos == 'FWD':
            for id in bench_ids:
                if id in df[(df['positions'] == 'FWD')]['id'].tolist():
                    return id

        # If no player is found, return None
        return None

    def single_gw_simulation(gw, df):
        # Initialise the gameweek points to 0.
        team_gameweek_points = 0

        # Ensures no points calculations are attempted for non-existent gameweeks.
        if gw > current_gw:
            return team_gameweek_points

        # Drop all columns with gw in that dont contain 'gw_{gw}'
        keep_columns = ['id', 'web_name', 'positions', f'gw_{gw}_points', f'gw_{gw}_minutes', 
                        'in_lineup', 'on_bench', 'is_captain', 'is_vice_captain']
        
        # Filter the dataframe to only include the specific gameweek columns. Also define the starting and
        # bench players, as well as the counts for starting positions. This is important in establishing
        # valid transfers. 
        df = df[keep_columns]
        starting_players = df[(df['in_lineup'] == True)]
        bench_player_ids = df[(df['on_bench'] == True)]['id'].tolist()
        starting_defenders = len(starting_players[(starting_players['positions'] == 'DEF')])
        starting_midfielders = len(starting_players[(starting_players['positions'] == 'MID')])
        starting_forwards = len(starting_players[(starting_players['positions'] == 'FWD')])

        # Add the gameweek points to a total for starting players, if they didn't play,
        # they are added to a list for attempted substitutions.
        player_not_played = []
        for index, row in starting_players.iterrows():
            if row[f'gw_{gw}_minutes'] > 0:
                team_gameweek_points += row[f'gw_{gw}_points']
            else:
                player_not_played.append(row['id'])
        
        # Captain points are added inherently in the original iterative statement, so this needs to be 
        # handled again to double the captain or vice captain's points. Flags indicate whether the captain
        # played in that gameweek, or whether the vc scored double points.
        captain = df[(df['is_captain'] == True)].iloc[0]
        vice_captain = df[(df['is_vice_captain'] == True)].iloc[0]
        did_captain_play = False
        vice_play_in_captains_place = False
        if captain[f'gw_{gw}_minutes'] > 0:
            team_gameweek_points += captain[f'gw_{gw}_points']
            did_captain_play = True
        else:
            team_gameweek_points += vice_captain[f'gw_{gw}_points']
            if vice_captain[f'gw_{gw}_minutes'] > 0:
                vice_play_in_captains_place = True
        
        subs_made = []
        # Attempts to sub each player.
        for player in player_not_played:
            player_position = df[(df['id'] == player)]['positions'].iloc[0]
            # Selection to enforce valid team formation logic for substitutions.
            if bench_player_ids is not None:
                if player_position == 'DEF':
                    if starting_defenders <= 3:
                        substitute_id = get_bench_player(df, bench_player_ids, pos='DEF')
                        if substitute_id != None:
                            bench_player_ids.remove(substitute_id)
                            starting_defenders += 1
                            team_gameweek_points += df[(df['id'] == substitute_id)][f'gw_{gw}_points'].iloc[0]
                            subs_made.append({'sub_out:': player, 'sub_in': substitute_id})
                    else:
                        substitute_id = get_bench_player(df, bench_player_ids, pos='*')
                        if substitute_id != None:
                            bench_player_ids.remove(substitute_id)
                            sub_position = df[(df['id'] == substitute_id)]['positions'].iloc[0]
                            if sub_position == 'DEF':
                                starting_defenders += 1
                            elif sub_position == 'MID':
                                starting_midfielders += 1
                            elif sub_position == 'FWD':
                                starting_forwards += 1
                            team_gameweek_points += df[(df['id'] == substitute_id)][f'gw_{gw}_points'].iloc[0]
                            subs_made.append({'sub_out:': player, 'sub_in': substitute_id})
                if player_position == 'MID':
                    if starting_midfielders <= 2:
                        substitute_id = get_bench_player(df, bench_player_ids, pos='MID')
                        if substitute_id != None:
                            bench_player_ids.remove(substitute_id)
                            starting_midfielders += 1
                            team_gameweek_points += df[(df['id'] == substitute_id)][f'gw_{gw}_points'].iloc[0]
                            subs_made.append({'sub_out:': player, 'sub_in': substitute_id})
                    else:
                        substitute_id = get_bench_player(df, bench_player_ids, pos='*')
                        if substitute_id != None:
                            bench_player_ids.remove(substitute_id)
                            sub_position = df[(df['id'] == substitute_id)]['positions'].iloc[0]
                            if sub_position == 'DEF':
                                starting_defenders += 1
                            elif sub_position == 'MID':
                                starting_midfielders += 1
                            elif sub_position == 'FWD':
                                starting_forwards += 1
                            team_gameweek_points += df[(df['id'] == substitute_id)][f'gw_{gw}_points'].iloc[0]
                            subs_made.append({'sub_out:': player, 'sub_in': substitute_id})
                if player_position == 'FWD':
                    if starting_forwards <= 1:
                        substitute_id = get_bench_player(df, bench_player_ids, pos='FWD')
                        if substitute_id != None:
                            bench_player_ids.remove(substitute_id)
                            starting_forwards += 1
                            team_gameweek_points += df[(df['id'] == substitute_id)][f'gw_{gw}_points'].iloc[0]
                            subs_made.append({'sub_out:': player, 'sub_in': substitute_id})
                    else:
                        substitute_id = get_bench_player(df, bench_player_ids, pos='*')
                        if substitute_id != None:
                            bench_player_ids.remove(substitute_id)
                            sub_position = df[(df['id'] == substitute_id)]['positions'].iloc[0]
                            if sub_position == 'DEF':
                                starting_defenders += 1
                            elif sub_position == 'MID':
                                starting_midfielders += 1
                            elif sub_position == 'FWD':
                                starting_forwards += 1
                            team_gameweek_points += df[(df['id'] == substitute_id)][f'gw_{gw}_points'].iloc[0]
                            subs_made.append({'sub_out:': player, 'sub_in': substitute_id})

        # Returns a dictionary for results of the simulation
        return_dic = {
            'points': team_gameweek_points,
            'subs_made': subs_made,
            'did_captain_play': did_captain_play,
            'vice_play_in_captains_place': vice_play_in_captains_place
        }
        return return_dic
    
    # Simulates the points the set and forget team would've scored across all gameweeks.
    points_by_gw = []
    for i in range(1, current_gw+1):
        gw_simulation_result = single_gw_simulation(i, model_players_df)
        gameweek_dic = {
            i : gw_simulation_result
        }
        points_by_gw.append(gameweek_dic)

    return points_by_gw

if __name__ == "__main__":
    min_gameweek = 1
    max_gameweek = 38
    player_gameweek_df = all_player_data(min_gameweek, max_gameweek)

    lineup, bench, captaincy, vice_captaincy = basic_set_and_forget(player_gameweek_df, 0.15, 1000)
    
    model_output = retrieve_refactored_model_output(lineup, bench, captaincy, vice_captaincy)
    model_players_df = retrieve_model_gameweek_history(model_output, player_gameweek_df)
    print(model_players_df)

    simulated_season = simulate_model_team(model_players_df)

    # Get total points
    total_points = 0
    for gw in simulated_season:
        for gw_num, gw_data in gw.items():
            total_points += gw_data['points']

    print("Total team points: ",total_points)