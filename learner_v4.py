from __future__ import print_function
import pyodbc
import os
import time
import numpy as np
import pickle
from sklearn.feature_extraction import FeatureHasher
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.svm import LinearSVC,SVC
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report, accuracy_score, make_scorer
from collections import defaultdict
from datetime import datetime, timedelta
import pandas as pd
import struct
import yaml
#from pymemcache.client.base import Client
import redis
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import model_selection, metrics   #Additional scklearn functions
from sklearn.model_selection import GridSearchCV   #Perforing grid search
import uuid
from utils import ODBC_DIRECTIVES


DB = os.getenv('DB', 'vwSeedionData')

def handle_datetimeoffset(dto_value):
    # ref: https://github.com/mkleehammer/pyodbc/issues/134#issuecomment-281739794
    tup = struct.unpack("<6hI2h", dto_value)  # e.g., (2017, 3, 16, 10, 35, 18, 0, -6, 0)
    tweaked = [tup[i] // 100 if i == 6 else tup[i] for i in range(len(tup))]
    return "{:04d}-{:02d}-{:02d} {:02d}:{:02d}:{:02d}.{:07d} {:+03d}:{:02d}".format(*tweaked)

cxn = pyodbc.connect(";".join(ODBC_DIRECTIVES))
cxn.setencoding(encoding='utf-16le')
cxn.add_output_converter(-155, handle_datetimeoffset)
cursor = cxn.cursor()
r = redis.StrictRedis(host='localhost', port=6379, db=0)

def get_team_features_sql(teamName, date):
    if os.getenv("WITH_PICKLE") == "1":
        return

    top = ""
    if os.getenv("TOP_N"):
        top = "TOP %s" % os.getenv("TOP_N")

    print('teamName: ', teamName)

    queryThisSeason = "select max(EventSeason) from %s where EventStartDate < '%s'" % (DB, date)
    cursor.execute(queryThisSeason)
    thisSeason = cursor.fetchall()[0][0]

    queryTotalWins = "select count(distinct GameId) from %s where GameTeamNameWinner='%s' and EventStartDate < '%s'" % (DB, teamName, date)
    cursor.execute(queryTotalWins)
    teamTotalWins = cursor.fetchall()[0][0]
    print('teamTotalWins: ', teamTotalWins)

    queryTotalLoses = "select count(distinct GameId) from %s where GameTeamNameLoser='%s' and EventStartDate < '%s'" % (DB, teamName, date)
    cursor.execute(queryTotalLoses)
    teamTotalLosses = cursor.fetchall()[0][0]
    print('teamTotalLosses: ', teamTotalLosses)

    if teamTotalWins == 0 and teamTotalLosses == 0:
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    queryTotalWinsThisSeason = "select count(distinct GameId) from %s where GameTeamNameWinner='%s' and EventSeason='%s' and EventStartDate < '%s'" % (DB, teamName, thisSeason, date)
    cursor.execute(queryTotalWinsThisSeason)
    teamTotalWinsThisSeason = cursor.fetchall()[0][0]
    print('teamTotalWinsThisSeason: ', teamTotalWinsThisSeason)

    queryTotalLosesThisSeason = "select count(distinct GameId) from %s where GameTeamNameLoser='%s' and EventSeason='%s' and EventStartDate < '%s'" % (DB, teamName, thisSeason, date)
    cursor.execute(queryTotalLosesThisSeason)
    teamTotalLosesThisSeason = cursor.fetchall()[0][0]
    print('teamTotalLosesThisSeason: ', teamTotalLosesThisSeason)

    queryTournamentsPlayedThisSeason = "select count(distinct EventName) from %s where TeamName='%s' and EventSeason='%s' and EventStartDate < '%s'" % (DB, teamName, thisSeason, date)
    cursor.execute(queryTournamentsPlayedThisSeason)
    teamTournamentsPlayedThisSeason = cursor.fetchall()[0][0]
    print('teamTournamentsPlayedThisSeason: ', teamTournamentsPlayedThisSeason)

    yearBefore = date - timedelta(days=365)
    queryTournamentsPlayed = "select count(distinct EventName) from %s where TeamName='%s' and EventStartDate < '%s'" % (DB, teamName, date)
    cursor.execute(queryTournamentsPlayed)
    teamTournamentsPlayed = cursor.fetchall()[0][0]
    print('teamTournamentsPlayed: ', teamTournamentsPlayed)

    queryTournamentsThisYear = "select distinct EventName from %s where TeamName='%s' and EventStartDate < '%s'" % (DB, teamName, date)
    cursor.execute(queryTournamentsThisYear)
    teamTournamentsThisYear = cursor.fetchall()

    games_num = 0
    if len(teamTournamentsThisYear) == 0:
        averageGamesPerTournament = 0
    else:
        for tournament in teamTournamentsThisYear:
            tournament = tournament[0]
            queryDistinctGames = "select count(distinct GameId) from %s where EventName='%s' and TeamName='%s' and EventStartDate < '%s'" % (DB, tournament, teamName, date)
            cursor.execute(queryDistinctGames)
            distinctGames = cursor.fetchall()[0][0]
            games_num += distinctGames
        averageGamesPerTournament = games_num / len(teamTournamentsThisYear)

    print('averageGamesPerTournament: ', averageGamesPerTournament)

    queryTournamentsWonThisYear = "select distinct EventName from %s where GameTeamNameWinner='%s' and EventStartDate < '%s'" % (DB, teamName, date)
    cursor.execute(queryTournamentsWonThisYear)
    teamTournamentsWonThisYear = cursor.fetchall()

    won_num = 0
    if len(teamTournamentsWonThisYear) == 0:
        averageWinsPerTournament = 0
    else:
        for tournament in teamTournamentsWonThisYear:
            tournament = tournament[0]
            #queryDistinctGames = "select count(distinct GameId) from %s where EventName='%s' and GameTeamNameWinner='%s' and EventStartDate < '%s'" % (DB, tournament, teamName, date)
            cursor.execute("select count(distinct GameId) from %s where EventName=? and GameTeamNameWinner=? and EventStartDate < ?" % (DB), (tournament, teamName, date,))
            distinctGames = cursor.fetchall()[0][0]
            won_num += distinctGames

        averageWinsPerTournament = won_num / len(teamTournamentsWonThisYear)

    print('averageWinsPerTournament: ', averageWinsPerTournament)

    queryDistinctWonGames = "select distinct GameId from %s where GameTeamNameWinner='%s' and EventStartDate < '%s'" % (DB, teamName, date)
    cursor.execute(queryDistinctWonGames)
    distinctWonGames = cursor.fetchall()

    quesryAveragePointsPerGame = "select avg(GameHomeTeamPoints) from %s where GameTeamNameWinner='%s' and GameTeamNameHome='%s' and EventStartDate < '%s'" % (DB, teamName, teamName, date)
    cursor.execute(quesryAveragePointsPerGame)
    averagePointsPerWinHome = cursor.fetchall()[0][0]

    quesryAveragePointsPerGame = "select avg(GameAwayTeamPoints) from %s where GameTeamNameWinner='%s' and GameTeamNameAway='%s' and EventStartDate < '%s'" % (DB, teamName, teamName, date)
    cursor.execute(quesryAveragePointsPerGame)
    averagePointsPerWinAway = cursor.fetchall()[0][0]

    if averagePointsPerWinHome == None:
        averagePointsPerWinHome = 0
    if averagePointsPerWinAway == None:
        averagePointsPerWinAway = 0

    averagePointsPerWin = (averagePointsPerWinHome + averagePointsPerWinAway) / 2
    print('averagePointsPerWin: ', averagePointsPerWin)

    quesryAveragePointsPerGame = "select avg(GameHomeTeamPoints) from %s where GameTeamNameLoser='%s' and GameTeamNameHome='%s' and EventStartDate < '%s'" % (DB, teamName, teamName, date)
    cursor.execute(quesryAveragePointsPerGame)
    averagePointsPerLoseHome = cursor.fetchall()[0][0]

    quesryAveragePointsPerGame = "select avg(GameAwayTeamPoints) from %s where GameTeamNameLoser='%s' and GameTeamNameAway='%s' and EventStartDate < '%s'" % (DB, teamName, teamName, date)
    cursor.execute(quesryAveragePointsPerGame)
    averagePointsPerLoseAway = cursor.fetchall()[0][0]

    if averagePointsPerLoseHome == None:
        averagePointsPerLoseHome = 0
    if averagePointsPerLoseAway == None:
        averagePointsPerLoseAway = 0

    averagePointsPerLose = (averagePointsPerLoseHome + averagePointsPerLoseAway) / 2
    print('averagePointsPerLose: ', averagePointsPerLose)

    queryLatestDate = "select max(EventStartDate) from %s where TeamName='%s' and EventStartDate < '%s'" % (DB, teamName, date)
    cursor.execute(queryLatestDate)
    lastDate = cursor.fetchall()[0][0]

    queryTeamPlayers = "select distinct PlayerId from %s where TeamName='%s' and EventStartDate = '%s'" % (DB, teamName, lastDate)
    cursor.execute(queryTeamPlayers)
    teamPlayers = cursor.fetchall()

    if len(teamPlayers) == 0:
        averagePlayersRankingPoints = 0
    else:
        player_ids = [teamPlayer[0] for teamPlayer in teamPlayers]
        all_points = 0
        for player_id in player_ids:
            queryPlayerRankingPoints = "select PlayerRankingPoints from %s where PlayerId = '%s'" % (DB, player_id)
            cursor.execute(queryPlayerRankingPoints)
            points = cursor.fetchall()[0][0]
            all_points += points

        averagePlayersRankingPoints = all_points / len(teamPlayers)

    print('averagePlayersRankingPoints: ', averagePlayersRankingPoints)
    print('========================================================================================================')
    #queryTeamFinalStanding = "select TeamFinalStanding from %s where TeamName=%s and EventStartDate < %" (DB, teamName, date)

    df = [teamTotalWins, teamTotalLosses, teamTotalWinsThisSeason, teamTotalLosesThisSeason, teamTournamentsPlayedThisSeason, teamTournamentsPlayed, averageGamesPerTournament,  averageWinsPerTournament, averagePointsPerWin, averagePointsPerLose, averagePlayersRankingPoints]

    return df

def get_team_features_v1(teamName, date=datetime.now(), days_before=365, data=None):
    startDate = date - timedelta(days=days_before)

    if data is None:
        queryAll = "select * from %s where EventStartDate > ? and EventStartDate < ? and (GameTeamNameHome = ? or GameTeamNameAway = ?)" % DB
        data = pd.read_sql(queryAll, cxn, params=[startDate, date, teamName, teamName])
        data.set_index(['GameId', 'PlayerId'])
        #data = data.drop_duplicates(['PlayerId'])

    thisSeason = data['EventSeason'].max()
    teamTotalWins = data[teamName == data['GameTeamNameWinner']]['GameId'].nunique()

    teamTotalLoses = data[teamName == data['GameTeamNameLoser']]['GameId'].nunique()

    if teamTotalWins == 0 and teamTotalLoses == 0:
        teamTotalWinsThisSeason = 0
        teamTotalLosesThisSeason = 0
        teamTournamentsPlayedThisSeason = 0
        teamTournamentsPlayed = 0
    else:

        teamTotalWinsThisSeason = data[(teamName == data['GameTeamNameWinner']) & (thisSeason == data['EventSeason'])]['GameId'].nunique()
        teamTotalLosesThisSeason = data[(teamName == data['GameTeamNameLoser']) & (thisSeason == data['EventSeason'])]['GameId'].nunique()
        teamTournamentsPlayedThisSeason = data[((teamName == data['GameTeamNameHome']) | (teamName == data['GameTeamNameAway'])) & (thisSeason == data['EventSeason'])]['EventName'].nunique()
        teamTournamentsPlayed = data[(teamName == data['GameTeamNameHome']) | (teamName == data['GameTeamNameAway'])]['EventName'].nunique()

    #averagePointsPerWinGame = (data[(teamName == data['GameTeamNameWinner']) & (teamName == data['GameTeamNameHome'])]['GameHomeTeamPoints'].mean() + \
    #    data[(teamName == data['GameTeamNameWinner']) & (teamName == data['GameTeamNameAway'])]['GameAwayTeamPoints'].mean()) / 2
    #print('averagePointsPerWin: ', averagePointsPerWinGame)

    #averagePointsPerLoseGame = (data[(teamName == data['GameTeamNameLoser']) & (teamName == data['GameTeamNameHome'])]['GameHomeTeamPoints'].mean() + \
    #    data[(teamName == data['GameTeamNameLoser']) & (teamName == data['GameTeamNameAway'])]['GameAwayTeamPoints'].mean()) / 2
    #print('averagePointsPerLose: ', averagePointsPerLoseGame)

    #averageWinsPerTournament = data[teamName == 'GameTeamNameWinner']

    last_date = data[(teamName == data['GameTeamNameHome']) | (teamName == data['GameTeamNameAway'])]['EventStartDate'].max()
    last_game_data = data[((teamName == data['GameTeamNameHome']) | (teamName == data['GameTeamNameAway'])) & (data['EventStartDate'] == last_date)]

    teamPlayerIds = last_game_data[(teamName == data['GameTeamNameHome']) | (teamName == data['GameTeamNameAway'])]['PlayerId'].value_counts().nlargest(4).keys()

    print('teamName: ', teamName)

    teamPlayers = last_game_data[last_game_data['PlayerId'].isin(teamPlayerIds)].drop_duplicates(['PlayerId'])[['PlayerId', 'PlayerFirstName', 'PlayerLastName', 'PlayerRankingPoints']]
    print("\nPlayers: ")
    print(teamPlayers)

    averageRankingPoints = teamPlayers['PlayerRankingPoints'].mean()
    if averageRankingPoints is np.nan:
        averageRankingPoints = 0

    print('\nteamTotalWins: ', teamTotalWins)
    print('teamTotalLoses: ', teamTotalLoses)
    print('teamTotalWinsThisSeason: ', teamTotalWinsThisSeason)
    print('teamTotalLosesThisSeason: ', teamTotalLosesThisSeason)
    print('teamTournamentsPlayedThisSeason: ', teamTournamentsPlayedThisSeason)
    print('teamTournamentsPlayed: ', teamTournamentsPlayed)
    print('averageRankingPoints: ', averageRankingPoints)
    print('========================================================================================================')

    return np.nan_to_num(np.asarray([teamTotalWins, teamTotalLoses, teamTotalWinsThisSeason, teamTotalLosesThisSeason, teamTournamentsPlayed, averageRankingPoints]))

#
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
#

def get_team_features(teamName, date=datetime.now(), days_before=365, data=None, game_id=None):
    startDate = date - timedelta(days=days_before)

    if data is None:
        #print(teamName)
        queryAll = "select * from %s where EventStartDate > ? and EventStartDate < ? and (GameTeamNameHome = ? or GameTeamNameAway = ?)" % DB
        data = pd.read_sql(queryAll, cxn, params=[startDate, date, teamName.encode('utf-16le'), teamName.encode('utf-16le')])
        data.set_index(['GameId', 'PlayerId'])
        #data = data.drop_duplicates(['PlayerId'])

    #print(data)
    #print(game_id)
    thisSeason = data['EventSeason'].max()
    teamTotalWins = data[teamName == data['GameTeamNameWinner']]['GameId'].nunique()

    teamTotalLoses = data[teamName == data['GameTeamNameLoser']]['GameId'].nunique()

    if teamTotalWins == 0 and teamTotalLoses == 0:
        teamTotalWinsThisSeason = 0
        teamTotalLosesThisSeason = 0
        teamTournamentsPlayedThisSeason = 0
        teamTournamentsPlayed = 0
    else:

        teamTotalWinsThisSeason = data[(teamName == data['GameTeamNameWinner']) & (thisSeason == data['EventSeason'])]['GameId'].nunique()
        teamTotalLosesThisSeason = data[(teamName == data['GameTeamNameLoser']) & (thisSeason == data['EventSeason'])]['GameId'].nunique()
        teamTournamentsPlayedThisSeason = data[((teamName == data['GameTeamNameHome']) | (teamName == data['GameTeamNameAway'])) & (thisSeason == data['EventSeason'])]['EventName'].nunique()
        teamTournamentsPlayed = data[(teamName == data['GameTeamNameHome']) | (teamName == data['GameTeamNameAway'])]['EventName'].nunique()

    #averagePointsPerWinGame = (data[(teamName == data['GameTeamNameWinner']) & (teamName == data['GameTeamNameHome'])]['GameHomeTeamPoints'].mean() + \
    #    data[(teamName == data['GameTeamNameWinner']) & (teamName == data['GameTeamNameAway'])]['GameAwayTeamPoints'].mean()) / 2
    #print('averagePointsPerWin: ', averagePointsPerWinGame)

    #averagePointsPerLoseGame = (data[(teamName == data['GameTeamNameLoser']) & (teamName == data['GameTeamNameHome'])]['GameHomeTeamPoints'].mean() + \
    #    data[(teamName == data['GameTeamNameLoser']) & (teamName == data['GameTeamNameAway'])]['GameAwayTeamPoints'].mean()) / 2
    #print('averagePointsPerLose: ', averagePointsPerLoseGame)

    #averageWinsPerTournament = data[teamName == 'GameTeamNameWinner']

    if game_id is None:
        last_date = data[(teamName == data['TeamName'])]['EventStartDate'].max()
        last_game_data = data[(teamName == data['TeamName']) & (data['EventStartDate'] == last_date)]

        teamPlayerIds = last_game_data['PlayerId'].value_counts().nlargest(4).keys()
    else:
        gameQuery = "select * from %s where GameId = ? and TeamName = ?" % DB
        game = pd.read_sql(gameQuery, cxn, params=[game_id, teamName.encode('utf-16')])
        #print("***************")
        #print(game)
        #print("***************")
        teamPlayerIds = game['PlayerId'].unique()
        #print(teamPlayerIds)

        last_game_data = game

    #print('\nteamName: ', teamName)

    teamPlayers = last_game_data[last_game_data['PlayerId'].isin(teamPlayerIds)].drop_duplicates(['PlayerId'])[['PlayerId', 'PlayerFirstName', 'PlayerLastName', 'PlayerRankingPoints']]
    #print("\nPlayers: ")
    #print(teamPlayers)

    averageRankingPoints = teamPlayers['PlayerRankingPoints'].mean()
    if averageRankingPoints is np.nan:
        averageRankingPoints = 0

    #print('\nteamTotalWins: ', teamTotalWins)
    #print('teamTotalLoses: ', teamTotalLoses)
    #print('teamTotalWinsThisSeason: ', teamTotalWinsThisSeason)
    #print('teamTotalLosesThisSeason: ', teamTotalLosesThisSeason)
    #print('teamTournamentsPlayedThisSeason: ', teamTournamentsPlayedThisSeason)
    #print('teamTournamentsPlayed: ', teamTournamentsPlayed)
    #print('averageRankingPoints: ', averageRankingPoints)
    #print('========================================================================================================')

    return np.nan_to_num(np.asarray([teamTotalWins, teamTotalLoses, teamTotalWinsThisSeason, teamTotalLosesThisSeason, teamTournamentsPlayed, averageRankingPoints]))

def get_future_data(date=datetime.now(), end_date=None):

    if end_date is not None:
        queryAll = "select GameId, EventName, EventStartDate, GameTeamNameHome, GameTeamNameAway, GameTeamNameWinner, PlayerFirstName, PlayerLastName, PlayerRankingPoints, TeamName from %s where EventStartDate > '%s' and EventStartDate <= '%s'" % (DB, date, end_date)

    else:
        queryAll = "select GameId, EventName, EventStartDate, GameTeamNameHome, GameTeamNameAway, GameTeamNameWinner from %s where EventStartDate > '%s'" % (DB, date)

    data = pd.read_sql(queryAll, cxn, parse_dates=["EventStartDate"])
    #data['EventStartDate'] = pd.to_datetime(data['EventStartDate'].str[:data['EventStartDate'].str.rindex(' ')])

    rows = data.drop_duplicates(['GameId'])
    rows.sort_values(by=['EventStartDate'])

    #print(rows.shape)

    return rows, data

def flatten(l):
    for el in l:
        try:
            yield flatten(el)
        except TypeError:
            yield el

def get_winners_by_game_ids(start_date, end_date, game_ids):
    #queryAll = "select GameId, GameTeamNameWinner from {} where GameId IN ({})".format(DB, ','.join('\'%s\'' * len(game_ids)))# % (*game_ids)
    queryAll = "select GameId, GameTeamNameWinner from %s where EventStartDate > '%s' and EventStartDate < '%s'" % (DB, start_date, end_date)

    #params = [uuid.UUID(game_id) for game_id in game_ids]
    data = pd.read_sql(queryAll, cxn)

    rows = data.drop_duplicates(['GameId'])

    winners = rows[(rows['GameId'].isin(game_ids)) & (rows["GameTeamNameWinner"] != '') & (rows["GameTeamNameWinner"] != None) & (~rows["GameTeamNameWinner"].str.isspace())]

    #print("Winners: ", winners)

    return winners


def prepare_features(date=None, days_before=365):
    startDate = date - timedelta(days=days_before)

    queryAll = "select * from %s where EventStartDate > '%s'and EventStartDate < '%s'" % (DB, startDate, date)
    data = pd.read_sql(queryAll, cxn)
    data.set_index(['GameId', 'PlayerId', 'TeamName'])
    #data = data.sort(['EventStartDate'], ascending=[0])

    rows = data.drop_duplicates(['GameId'])[['GameTeamNameWinner', 'GameTeamNameLoser']]
    print(rows.shape)

    features = {}
    X_train = []
    y_train = []
    i = 0
    for index, row in rows.iterrows():
        #print(teams)
        #row = data[data['GameId'] == game].unique()
        #print(row)
        winner, loser = row['GameTeamNameWinner'], row['GameTeamNameLoser']
        #print('Two teams: ', winner, ' ', loser)

        if winner in features.keys():
            wtf = features[winner]
        else:
            wtf = get_team_features(winner, date, days_before, data, None)
            features[winner] = wtf
            #r.set(str(i), winner + '_1')
            #r.set(winner + '_1', wtf)
            i += 1

        if loser in features.keys():
            ltf = features[loser]
        else:
            ltf = get_team_features(loser, date, days_before, data, None)
            features[loser] = ltf
            #r.set(str(i), loser + '_1')
            #r.set(loser + '_1', ltf)
            i += 1

        X_train.append(np.concatenate([wtf, ltf]))
        y_train.append(1)
        #print(len(X_train))
        #print(X_train[len(X_train) - 1])

        X_train.append(np.concatenate([ltf, wtf]))
        y_train.append(0)
        print(len(X_train))
        print(len(y_train))
        y_t = np.asarray(y_train)
        print("Ones: ", len(y_t[y_t==1]))
        print("Zeros: ", len(y_t[y_t==0]))
        #print(X_train[len(X_train) - 1])

        if date == None:
            suffix = ''
        else:
            suffix = '_'+date.strftime('%d-%m-%Y')

        if len(y_train) % 1000 == 0:
            r.set("X_train" + suffix, np.asarray(X_train).ravel().tostring())
            r.set("y_train" + suffix, np.asarray(y_train).tostring())
        #print(i)
        #if i == 10:
        #    break

    #for i, (k, v) in enumerate(features.items()):
    #    r.set(str(i), k)
    #    r.set(k, v)

    #r.set("n_teams", len(features))

    #X_train = np.asarray(X_train)
    #y_train = np.asarray(y_train)

    #r.set("X", X_train.ravel().tostring())
    #r.set("y", y_train.ravel().tostring())

    return features, X_train, y_train

def load_features(num_columns=6, date=None):

    if date == None:
        suffix = '_21-04-2018'
    else:
        suffix = '_'+date.strftime('%d-%m-%Y')

    X_train_redis = r.get("X_train" + suffix)
    y_train_redis = r.get("y_train" + suffix)
    train_y = np.fromstring(y_train_redis, dtype=np.int64)
    #print(train_y.shape)
    train_x = np.fromstring(X_train_redis, dtype=np.float64).reshape(-1, num_columns)
    #print(train_x.shape)
    #print(len(train_y[train_y==0]))
    #print(len(train_y[train_y==1]))
    #indexes = np.random.choice(len(train_y), 60000)

    #X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.33, random_state=42)
    #X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, test_size=0.33, random_state=42)

    return train_x, train_y


def get_classifier(name):
    if name == "MONTE_CARLO":
        return SGDClassifier(verbose=1)
    if name == "RANDOM_FOREST":
        return RandomForestClassifier(verbose=1)
    if name == "NAIVE_BAYES":
        return GaussianNB()
    if name == "MAXENT":
        return LogisticRegression(verbose=1)
    if name == "NEURAL_NET":
        return MLPClassifier(solver='lbfgs', alpha=1e-5,
                             hidden_layer_sizes=(5, 2), random_state=1)
    if name == "XGB":
        return xgBoost()

    return SVC(random_state=13, verbose=1, kernel='rbf', probability=True)

def main():
    global originalclass, predictedclass
    start = time.time()
    print("Getting and preparing the data...")
    #features, X, y = prepare_fetures(datetime.today())
    X, y = load_features()
    print(X.shape)
    print(y.shape)
    took1 = time.time() - start
    print("Got data, took %.2f seconds" % took1)

    print("Building models with 10-fold cross-validation (for now)")

    classifiers = ['SVM'
        #'MONTE_CARLO', 'NAIVE_BAYES', 'MAXENT', 'SVM'
    ]
    # skipping 'RANDOM_FOREST' since it takes so long
    if os.getenv("CLASSIFIER"):
        classifiers = [os.getenv("CLASSIFIER")]

    for classifier in classifiers:
        print("========================")
        print("USING %s" % classifier)
        scores = cross_validate(
            get_classifier(classifier), X, y, scoring=['f1', 'precision', 'recall'],
            cv=10, verbose=1, n_jobs=8)
        for score_key in ['fit_time', 'score_time', 'test_f1', 'test_precision', 'test_recall']:
            print(score_key, "\t", np.mean(scores[score_key]))
        print("========================")
    took3 = time.time() - start
    print("Took %.2f seconds total" % took3)

def load_data():
    date = datetime.now()
    startDate = date - timedelta(days=1000)

    queryAll = "select * from %s where EventStartDate > '%s'and EventStartDate < '%s'" % (DB, startDate, date)
    data = pd.read_sql(queryAll, cxn)
    data.set_index(['GameId', 'PlayerId', 'TeamName'])
    #data = data.sort(['EventStartDate'], ascending=[0])

    rows = data.drop_duplicates(['GameId'])
    r.set("data", rows)

def save_classifier(name):
    clf = get_classifier(name)
    #X, y = load_features()
    #clf.fit(X, y)
    joblib.dump(clf, name + '_v1.pkl')


def modelfit(alg, dtrain, y, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):

    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain, label=y)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='auc', early_stopping_rounds=early_stopping_rounds, verbose_eval=True)
        alg.set_params(n_estimators=cvresult.shape[0])

    #Fit the algorithm on the data
    alg.fit(dtrain, y, eval_metric='auc')

    #Predict training set:
    dtrain_predictions = alg.predict(dtrain)
    dtrain_predprob = alg.predict_proba(dtrain)[:,1]

    #Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(y, dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(y, dtrain_predprob))

    #feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    #print(feat_imp)
    #feat_imp.plot(kind='bar', title='Feature Importances')
    #plt.ylabel('Feature Importance Score')

    return alg

def xgBoost():
    X, y = load_features()
    xgb1 = XGBClassifier(
     learning_rate =0.1,
     n_estimators=1000,
     max_depth=5,
     min_child_weight=1,
     gamma=0,
     subsample=0.8,
     colsample_bytree=0.8,
     objective= 'binary:logistic',
     nthread=4,
     scale_pos_weight=1,
     seed=27)

    xgb = modelfit(xgb1, X, y)

    return xgb

if __name__ == '__main__':
    #save_classifier("XGB")
    main()
    #prepare_features(datetime.now(), 500)
    #xgBoost()
    #X, y = load_features()
    #print(len(y[y==0]))

    #np.savez('Xy.npz', X=X, y=y)
    #data = np.load('Xy.npz')
    #print(data['X'].shape)
    #print(data['y'].shape)

    #r.set("X_train", data['X'].ravel().tostring())
    #r.set("y_train", data['y'].ravel().tostring())




