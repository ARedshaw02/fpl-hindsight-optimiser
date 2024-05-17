from fpl_data_retrieval import get_fpl_data, get_current_gameweek, player_gameweek_data
import pulp as plp
import pandas as pd
import numpy as np

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

def basic_set_and_forget(player_gameweek_df, bench_multiplier, budget):
    df = player_gameweek_df[["id", "total_points", "short_name", "positions", "start_cost"]]   
    
    players = df["id"]

    # Set up the problem
    model = plp.LpProblem("basic-set-forget", plp.LpMaximize)

    # Define the decision variables
    decisions = [
        plp.LpVariable(f"decision_{player}", lowBound=0, upBound=1, cat="Integer")
        for player in players
    ]
    captaincy = [
        plp.LpVariable(f"captaincy_{player}", lowBound=0, upBound=1, cat="Integer")
        for player in players
    ]
    bench_decisions = [
        plp.LpVariable(f"vice_captaincy_{player}", lowBound=0, upBound=1, cat="Integer")
        for player in players
    ]

    # Set the objective function
    model += sum(captaincy[i] + decisions[i] + (bench_multiplier * bench_decisions[i]) for i in range(len(players)))

    # Budget constraint
    model += sum(decisions[i] + bench_decisions[i] * df['start_cost'][i] for i in range(len(players))) <= budget

    # GK constraints
    model += sum(decisions[i] for i in range(len(players)) if df['positions'][i] == 'GK') == 1
    model += sum(decisions[i] + bench_decisions[i] for i in range(len(players)) if df['positions'][i] == 'GK') == 2

    # DEF constraints
    model += sum(decisions[i] for i in range(len(players)) if df['positions'][i] == 'DEF') >= 3
    model += sum(decisions[i] + bench_decisions[i] for i in range(len(players)) if df['positions'][i] == 'DEF') == 5

    # MID constraints
    model += sum(decisions[i] for i in range(len(players)) if df['positions'][i] == 'MID') >= 2
    model += sum(decisions[i] + bench_decisions[i] for i in range(len(players)) if df['positions'][i] == 'MID') == 5

    # FWD constraints
    model += sum(decisions[i] for i in range(len(players)) if df['positions'][i] == 'FWD') >= 1
    model += sum(decisions[i] + bench_decisions[i] for i in range(len(players)) if df['positions'][i] == 'FWD') == 3

    # Team constraints
    model += sum(decisions) == 11
    model += sum(decisions + bench_decisions) == 15
    model += sum(captaincy) == 1
    for teams in df['short_name'].unique():
        model += sum(decisions[i] + bench_decisions[i] for i in range(len(players)) if df['short_name'][i] == teams) <= 3
    
    for i in range(len(players)):
        model += (decisions[i] + bench_decisions[i]) >= 1
        model += (decisions[i] - captaincy[i]) >= 0

    plp.LpSolverDefault.msg = 0
    model.solve()

    return decisions, bench_decisions, captaincy

def select_team(expected_scores, prices, positions, clubs, total_budget=1000, sub_factor=0.2):
    num_players = len(expected_scores)
    model = plp.LpProblem("Constrained value maximisation", plp.LpMaximize)
    decisions = [
        plp.LpVariable("x{}".format(i), lowBound=0, upBound=1, cat='Integer')
        for i in range(num_players)
    ]
    captain_decisions = [
        plp.LpVariable("y{}".format(i), lowBound=0, upBound=1, cat='Integer')
        for i in range(num_players)
    ]
    sub_decisions = [
        plp.LpVariable("z{}".format(i), lowBound=0, upBound=1, cat='Integer')
        for i in range(num_players)
    ]


    # objective function:
    model += sum((captain_decisions[i] + decisions[i] + sub_decisions[i]*sub_factor) * expected_scores[i]
                 for i in range(num_players)), "Objective"

    # cost constraint
    model += sum((decisions[i] + sub_decisions[i]) * prices[i] for i in range(num_players)) <= total_budget  # total cost

    # position constraints
    # 1 starting goalkeeper
    model += sum(decisions[i] for i in range(num_players) if positions[i] == 'GK') == 1
    # 2 total goalkeepers
    model += sum(decisions[i] + sub_decisions[i] for i in range(num_players) if positions[i] == 'GK') == 2

    # 3-5 starting defenders
    model += sum(decisions[i] for i in range(num_players) if positions[i] == 'DEF') >= 3
    model += sum(decisions[i] for i in range(num_players) if positions[i] == 'DEF') <= 5
    # 5 total defenders
    model += sum(decisions[i] + sub_decisions[i] for i in range(num_players) if positions[i] == 'DEF') == 5

    # 3-5 starting midfielders
    model += sum(decisions[i] for i in range(num_players) if positions[i] == 'MID') >= 3
    model += sum(decisions[i] for i in range(num_players) if positions[i] == 'MID') <= 5
    # 5 total midfielders
    model += sum(decisions[i] + sub_decisions[i] for i in range(num_players) if positions[i] == 'MID') == 5

    # 1-3 starting attackers
    model += sum(decisions[i] for i in range(num_players) if positions[i] == 'FWD') >= 1
    model += sum(decisions[i] for i in range(num_players) if positions[i] == 'FWD') <= 3
    # 3 total attackers
    model += sum(decisions[i] + sub_decisions[i] for i in range(num_players) if positions[i] == 'FWD') == 3

    # club constraint
    for club_id in np.unique(clubs):
        model += sum(decisions[i] + sub_decisions[i] for i in range(num_players) if clubs[i] == club_id) <= 3  # max 3 players

    model += sum(decisions) == 11  # total team size
    model += sum(captain_decisions) == 1  # 1 captain
    
    for i in range(num_players):  
        model += (decisions[i] - captain_decisions[i]) >= 0  # captain must also be on team
        model += (decisions[i] + sub_decisions[i]) <= 1  # subs must not be on team

    model.solve()
    print("Total expected score = {}".format(model.objective.value()))

    return decisions, captain_decisions, sub_decisions

if __name__ == "__main__":
    min_gameweek = 1
    max_gameweek = 38
    player_gameweek_df = all_player_data(min_gameweek, max_gameweek)

    scores = player_gameweek_df["total_points"].values
    prices = player_gameweek_df["start_cost"].values
    positions = player_gameweek_df["positions"].values
    clubs = player_gameweek_df["short_name"].values

    decisions, captain_decisions, sub_decisions = select_team(scores, prices, positions, clubs)

    budget_spent = 0
    total_score = 0
    for decision in decisions:
        if decision.value() == 1:
            value = int(decision.name[1:])
            row = player_gameweek_df.iloc[value]
            print('STARTING: ',row['web_name'], '-', row['total_points'])
            total_score += row['total_points']
            budget_spent += row['start_cost']

    for decision in sub_decisions:
        if decision.value() == 1:
            value = int(decision.name[1:])
            row = player_gameweek_df.iloc[value]
            print('BENCH: ',row['web_name'], '-', row['total_points'])
            budget_spent += row['start_cost']

    for decision in captain_decisions:
        if decision.value() == 1:
            value = int(decision.name[1:])
            row = player_gameweek_df.iloc[value]
            total_score += row['total_points']
            print('CAPTAIN: ',row['web_name'], '-', row['total_points'])

    print('Total points:', total_score)
    print('Budget spent: ', budget_spent)
