from __future__ import print_function
import pyodbc
import os
import time
import numpy as np
import pickle
from sklearn.feature_extraction import FeatureHasher
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report, accuracy_score, make_scorer
from collections import defaultdict
from utils import ODBC_DIRECTIVES


PLAYER_GAME_TEAM = defaultdict(dict)
PLAYER_TEAM_GAME = defaultdict(dict)
TEAM_GAME_PLAYERS = defaultdict(dict)
GAME_TEAMS_PLAYERS = defaultdict(dict)
GAME_TO_WINNING_TEAM = {}
GAME_TO_HOME_TEAM = {}
GAME_TO_TEAM_POINTS = defaultdict(dict)
GAME_TO_DIVISION = {}
GAME_TO_GENDER = {}

DB = os.getenv('DB', 'vwSeedionData')

if os.getenv("WITH_PICKLE") == "1":
    print("Unpickling pre-processed feature data from features_%s.pickle" % DB)
    pickled = pickle.load(open('features_%s.pickle' % DB, 'rb'))
    PLAYER_GAME_TEAM = pickled['PLAYER_GAME_TEAM']
    PLAYER_TEAM_GAME = pickled['PLAYER_TEAM_GAME']
    TEAM_GAME_PLAYERS = pickled['TEAM_GAME_PLAYERS']
    GAME_TEAMS_PLAYERS = pickled['GAME_TEAMS_PLAYERS']
    GAME_TO_WINNING_TEAM = pickled['GAME_TO_WINNING_TEAM']
    GAME_TO_HOME_TEAM = pickled['GAME_TO_HOME_TEAM']
    GAME_TO_TEAM_POINTS = pickled['GAME_TO_TEAM_POINTS']
    GAME_TO_DIVISION = pickled['GAME_TO_DIVISION']
    GAME_TO_GENDER = pickled['GAME_TO_GENDER']

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
            ELSE 0
        END as did_win,
        GamePlayedAsHomeTeam,
        GameHomeTeamPoints,
        GameAwayTeamPoints,
        DivisionName,
        DivisionGender
    FROM %s
    """ % (top, DB)
    cxn = pyodbc.connect(";".join(ODBC_DIRECTIVES))
    cursor = cxn.cursor()
    cursor.execute(query)
    playerToWins = defaultdict(int)
    row = cursor.fetchone()
    while row:
        (player, team, gameid, did_win, was_home_team,
         home_points, away_points, division_name, division_gender) = row
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
        GAME_TO_DIVISION[gameid] = division_name
        GAME_TO_GENDER[gameid] = division_gender
        row = cursor.fetchone()

    with open('features_%s.pickle' % DB, 'wb') as fl:
        pickled = {}
        pickled['PLAYER_GAME_TEAM'] = PLAYER_GAME_TEAM
        pickled['PLAYER_TEAM_GAME'] = PLAYER_TEAM_GAME
        pickled['TEAM_GAME_PLAYERS'] = TEAM_GAME_PLAYERS
        pickled['GAME_TEAMS_PLAYERS'] = GAME_TEAMS_PLAYERS
        pickled['GAME_TO_WINNING_TEAM'] = GAME_TO_WINNING_TEAM
        pickled['GAME_TO_HOME_TEAM'] = GAME_TO_HOME_TEAM
        pickled['GAME_TO_TEAM_POINTS'] = GAME_TO_TEAM_POINTS
        pickled['GAME_TO_DIVISION'] = GAME_TO_DIVISION
        pickled['GAME_TO_GENDER'] = GAME_TO_GENDER
        pickle.dump(pickled, fl)

def game_features(game, team):
    for curr_team, players in GAME_TEAMS_PLAYERS[game].items():
        if team == curr_team:
            yield ("team_%s" % team, 1)
            for player in players:
                yield ("team_player_%s" % player, 1)
        else:
            yield ("opponent_%s" % team, 1)
            for player in players:
                yield ("opponent_player_%s" % player, 1)
    yield ("was_home", int(GAME_TO_HOME_TEAM.get(game, None) == team))
    yield ("division_%s" % GAME_TO_DIVISION[game], 1)
    yield ("gender_%s" % GAME_TO_GENDER[game], 1)


def build_features_and_classes():
    games_ordered = sorted(GAME_TO_WINNING_TEAM.keys())
    game_teams_sorted = []
    for game in games_ordered:
        for team in sorted(GAME_TO_TEAM_POINTS[game].keys()):
            game_teams_sorted.append((game, team))
    raw_X = [game_features(game, team) for game, team in game_teams_sorted]
    raw_Y = [int(team == GAME_TO_WINNING_TEAM[game]) for game, team in game_teams_sorted]
    hasher = FeatureHasher(input_type='pair', alternate_sign=False)
    X_features = hasher.transform(raw_X)

    if os.getenv("WITH_MONTE_CARLO") == "1":
         X_features = rbf_feature.fit_transform(X_features)

    return X_features, raw_Y


def accumulate_scoring(y_true, y_pred, **kwargs):
    print(classification_report(y_true, y_pred))
    return 0 # only in it for the classificaiton report

def get_classifier(name):
    if name == "MONTE_CARLO":
        return SGDClassifier(verbose=1)
    if name == "RANDOM_FOREST":
        return RandomForestClassifier(verbose=1)
    if name == "NAIVE_BAYES":
        return MultinomialNB()
    if name == "MAXENT":
        return LogisticRegression(verbose=1)
    if name == "NEURAL_NET":
        return MLPClassifier(solver='lbfgs', alpha=1e-5,
                             hidden_layer_sizes=(5, 2), random_state=1)
    return LinearSVC(random_state=0, verbose=1)

def main():
    global originalclass, predictedclass
    start = time.time()
    print("Getting data...")
    get_features()
    took1 = time.time() - start
    print("Got data, took %.2f seconds" % took1)
    print("Quick report on features, etc:")
    print("%d teams" % len(TEAM_GAME_PLAYERS.keys()))
    print("%d players" % len(PLAYER_GAME_TEAM.keys()))
    print("%d games" % len(GAME_TO_HOME_TEAM.keys()))
    print("Vectorizing features and classes...")
    build_feature_start = time.time()
    X, y = build_features_and_classes()
    took2 = time.time() - build_feature_start
    print("Got data, took %.2f seconds" % took2)
    print("Building models with 10-fold cross-validation (for now)")

    classifiers = [
        'MONTE_CARLO', 'NAIVE_BAYES', 'MAXENT', 'SVM'
    ]
    # skipping 'RANDOM_FOREST' since it takes so long
    if os.getenv("CLASSIFIER"):
        classifiers = [os.getenv("CLASSIFIER")]

    for classifier in classifiers:
        print("========================")
        print("USING %s" % classifier)
        scores = cross_validate(
            get_classifier(classifier), X, y, scoring=['f1', 'precision', 'recall'],
            cv=10, verbose=1, n_jobs=int(os.getenv("NUM_JOBS", "1")))
        for score_key in ['fit_time', 'score_time', 'test_f1', 'test_precision', 'test_recall']:
            print(score_key, "\t", np.mean(scores[score_key]))
        print("========================")
    took3 = time.time() - start
    print("Took %.2f seconds total" % took3)

if __name__ == '__main__':
    main()
