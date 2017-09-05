from __future__ import print_function
import pyodbc
import os
from collections import defaultdict
from time import time

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

PLAYER_GAME_POINTS = defaultdict(dict)
PLAYER_GAME_TEAM = defaultdict(dict)
PLAYER_TEAM_GAME = defaultdict(dict)
TEAM_GAME_PLAYERS = defaultdict(dict)
GAME_TEAMS_PLAYERS = defaultdict(dict)
GAME_TO_WINNING_TEAM = {}

def get_connection():
    cxn = pyodbc.connect(";".join(ODBC_DIRECTIVES))
    return connection

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
        EventStartDate,
        CASE WHEN (GamePlayedAsHomeTeam = 1 AND GameWonBy = 'Home')
            OR (GamePlayedAsHomeTeam != 1 AND GameWonBy = 'Away')
            THEN 1
            ELSE -1
        END
    FROM vwSeedionData
    """ % top
    cursor = get_connection().cursor()
    cursor.execute(query)
    playerToWins = defaultdict(int)
    row = cursor.fetchone()
    while row:
        (player, team, game, date, point) = row
        gameid = "%s-%s" % (game, date)
        PLAYER_GAME_POINTS[player][gameid] = point
        PLAYER_GAME_TEAM[player][gameid] = team
        PLAYER_TEAM_GAME[player][team] = gameid
        TEAM_GAME_PLAYERS[team][game] = set(list(TEAM_GAME_PLAYERS[team].get(game, [])) + [player])
        GAME_TEAMS_PLAYERS[game][team] = set(list(GAME_TEAM_PLAYERS[game].get(team, [])) + [player])
        if point == 1:
            GAME_TO_WINNING_TEAM[gameid] = team
        row = cursor.fetchone()

def main():
    start = time.time()
    print("Getting data...")
    get_features()
    took = time.time() - start
    print("Got data, took %.2f seconds" % took)


if __name__ == '__main__':
    main()
