from __future__ import print_function
import pyodbc
import os
import time
import pickle
from collections import defaultdict
from utils import ODBC_DIRECTIVES


PLAYER_GAME_POINTS = defaultdict(dict)
PLAYER_GAME_TEAM = defaultdict(dict)
PLAYER_TEAM_GAME = defaultdict(dict)
TEAM_GAME_PLAYERS = defaultdict(dict)
GAME_TEAMS_PLAYERS = defaultdict(dict)
GAME_TO_WINNING_TEAM = {}

DB = os.getenv('DB', 'vwSeedionData')


if os.getenv("WITH_PICKLE") == "1":
    print("Unpickling pre-processed feature data from heuristic_features_%s.pickle" % DB)
    pickled = pickle.load(open('heuristic_features_%s.pickle' % DB, 'rb'))
    PLAYER_GAME_POINTS = pickled['PLAYER_GAME_POINTS']
    PLAYER_GAME_TEAM = pickled['PLAYER_GAME_TEAM']
    PLAYER_TEAM_GAME = pickled['PLAYER_TEAM_GAME']
    TEAM_GAME_PLAYERS = pickled['TEAM_GAME_PLAYERS']
    GAME_TEAMS_PLAYERS = pickled['GAME_TEAMS_PLAYERS']
    GAME_TO_WINNING_TEAM = pickled['GAME_TO_WINNING_TEAM']

def get_features():
    if os.getenv("WITH_PICKLE") == "1":
        return
    top = ""
    if os.getenv("TOP_N"):
        top = "TOP %s" % os.getenv("TOP_N")
    query = """
    SELECT
        %s
        PlayerId,
        TeamName,
        GameId,
        CASE WHEN (GamePlayedAsHomeTeam = 1 AND GameWonBy = 'Home')
            OR (GamePlayedAsHomeTeam != 1 AND GameWonBy = 'Away')
            THEN 1
            ELSE -1
        END
    FROM %s
    """ % (top, DB)
    cxn = pyodbc.connect(";".join(ODBC_DIRECTIVES))
    cursor = cxn.cursor()
    cursor.execute(query)
    playerToWins = defaultdict(int)
    row = cursor.fetchone()
    while row:
        (player, team, gameid, point) = row
        PLAYER_GAME_POINTS[player][gameid] = point
        PLAYER_GAME_TEAM[player][gameid] = team
        PLAYER_TEAM_GAME[player][team] = gameid
        TEAM_GAME_PLAYERS[team][gameid] = set(list(TEAM_GAME_PLAYERS[team].get(gameid, [])) + [player])
        GAME_TEAMS_PLAYERS[gameid][team] = set(list(GAME_TEAMS_PLAYERS[gameid].get(team, [])) + [player])
        if point == 1:
            GAME_TO_WINNING_TEAM[gameid] = team
        row = cursor.fetchone()
    with open('heuristic_features_%s.pickle' % DB, 'wb') as fl:
        pickled = {}
        pickled['PLAYER_GAME_POINTS'] = PLAYER_GAME_POINTS
        pickled['PLAYER_GAME_TEAM'] = PLAYER_GAME_TEAM
        pickled['PLAYER_TEAM_GAME'] = PLAYER_TEAM_GAME
        pickled['TEAM_GAME_PLAYERS'] = TEAM_GAME_PLAYERS
        pickled['GAME_TEAMS_PLAYERS'] = GAME_TEAMS_PLAYERS
        pickled['GAME_TO_WINNING_TEAM'] = GAME_TO_WINNING_TEAM
        pickle.dump(pickled, fl)


def main():
    start = time.time()
    print("Getting data...")
    get_features()
    took = time.time() - start
    print("Got data, took %.2f seconds" % took)
    tp = 0.0
    tn = 0.0
    fp = 0.0
    fn = 0.0
    for game in GAME_TEAMS_PLAYERS.keys():
        scorer = []
        for team in GAME_TEAMS_PLAYERS[game].keys():
            total_score = 0
            for player in GAME_TEAMS_PLAYERS[game][team]:
                # lazy leave-one-out validation
                total_score += sum(PLAYER_GAME_POINTS[player].values())
                total_score -= PLAYER_GAME_POINTS[player][game]
        scorer.append((team, total_score))
        predicted_winner = sorted(scorer, reverse=True, key=lambda x: x[1])[0][0]
        if predicted_winner == GAME_TO_WINNING_TEAM.get(game, None):
            # correctly predicted the winner and the loser
            tp += 1.0
            tn += 1.0
        else:
            # incorreclty predicted the winner and the loser
            fp += 1.0
            fn += 1.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    true_negative_rate = tn / (tn + fp)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    fscore = 2 * ((precision * recall) / (precision + recall))
    print("""
Precision\t%.2f
Recall\t%.2f
True Neg\t%.2f
Acc\t%.2f
Fscore\t%.2f
          """ % (precision, recall, true_negative_rate, accuracy, fscore))
    overall_time = time.time() - start
    print("Finished in %.2f seconds" % overall_time)

if __name__ == '__main__':
    main()
