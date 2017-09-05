from __future__ import print_function
import pyodbc
import os
import time
import numpy as np
from sklearn.feature_extraction import FeatureHasher
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from collections import defaultdict

"""
Usually you don't store this information in version control,
but this is a private project with little variation.
"""
ODBC_DIRECTIVES = [
    "DRIVER={ODBC Driver 13 for SQL Server}",
    "PORT=1433",
    "SERVER=fiba3x3.database.windows.net",
    "DATABASE=FIBA_3x3",
    "UID=client_seedion",
    "PWD=BQcACAADBgwACAcHBA4MBQ",
]

PLAYER_GAME_TEAM = defaultdict(dict)
PLAYER_TEAM_GAME = defaultdict(dict)
TEAM_GAME_PLAYERS = defaultdict(dict)
GAME_TEAMS_PLAYERS = defaultdict(dict)
GAME_TO_WINNING_TEAM = {}
GAME_TO_HOME_TEAM = {}
GAME_TO_TEAM_POINTS = defaultdict(dict)

def get_features():
    top = ""
    if os.getenv("TOP_N"):
        top = "TOP %s" % os.getenv("TOP_N")
    query = """
    SELECT
        %s
        PlayerId,
        TeamName,
        GameName,
        convert(varchar, EventStartDate),
        CASE WHEN (GamePlayedAsHomeTeam = 1 AND GameWonBy = 'Home')
            OR (GamePlayedAsHomeTeam != 1 AND GameWonBy = 'Away')
            THEN 1
            ELSE 0
        END as did_win,
        GamePlayedAsHomeTeam,
        GameHomeTeamPoints,
        GameAwayTeamPoints
    FROM vwSeedionData
    """ % top
    cxn = pyodbc.connect(";".join(ODBC_DIRECTIVES))
    cursor = cxn.cursor()
    cursor.execute(query)
    playerToWins = defaultdict(int)
    row = cursor.fetchone()
    while row:
        (player, team, game, date, did_win, was_home_team, home_points, away_points) = row
        gameid = "%s-%s" % (game, date)
        PLAYER_GAME_TEAM[player][gameid] = team
        PLAYER_TEAM_GAME[player][team] = gameid
        TEAM_GAME_PLAYERS[team][gameid] = set(list(TEAM_GAME_PLAYERS[team].get(gameid, [])) + [player])
        GAME_TEAMS_PLAYERS[gameid][team] = set(list(GAME_TEAMS_PLAYERS[gameid].get(team, [])) + [player])
        if did_win:
            GAME_TO_WINNING_TEAM[gameid] = team
        if was_home_team:
            GAME_TO_TEAM_POINTS[gameid][team] = home_points
            GAME_TO_HOME_TEAM[gameid] = team
        else:
            GAME_TO_TEAM_POINTS[gameid][team] = away_points
        row = cursor.fetchone()

def game_features(game, team):
    for curr_team, players in GAME_TEAMS_PLAYERS[game].items():
        if team == curr_team:
            yield ("team", team)
            for player in players:
                yield ("team_player", player)
        else:
            yield ("opponent", team)
            for player in players:
                yield ("opponent_player", player)
    yield ("was_home", int(GAME_TO_HOME_TEAM[game] == team))

def build_features_and_classes():
    games_ordered = sorted(GAME_TO_WINNING_TEAM.keys())
    game_teams_sorted = []
    for game in games_ordered:
        for team in sorted(GAME_TO_TEAM_POINTS[game].keys()):
            game_teams_sorted.append((game, team))
    raw_X = [game_features(game, team) for game, team in game_teams_sorted]
    raw_Y = np.array([int(team == GAME_TO_WINNING_TEAM[game]) for game, team in game_teams_sorted], ndim=1)
    hasher = FeatureHasher(input_type='pair')
    return hasher.fit_transform(raw_X, raw_Y).toarray(), raw_Y


def main():
    start = time.time()
    print("Getting data...")
    get_features()
    took1 = time.time() - start
    print("Got data, took %.2f seconds" % took1)
    print("Vectorizing features and classes...")
    X, y = build_features_and_classes()
    took2 = time.time() - took1
    print("Got data, took %.2f seconds" % took2)
    print("Building models with 10-fold cross-validation")
    clf = LinearSVC(random_state=0)
    # todo: add precision, recall using custom scorer
    print(X)
    print(y)
    print(cross_val_score(clf, X, y, scoring='f1', groups=10, n_jobs=-1))
    took3 = time.time() - took2
    print("Got data, took %.2f seconds" % took3)

if __name__ == '__main__':
    main()
