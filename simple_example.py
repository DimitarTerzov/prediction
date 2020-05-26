import sys
import os
from datetime import datetime
import time
import logging
import csv
import pandas as pd
import numpy as np
import requests
import json
import traceback
from sklearn.externals import joblib
#import joblib
#import pickle


_XGB_CLASSIFIER = joblib.load(
    os.path.join(os.path.abspath(os.path.dirname(__file__)), 'XGB_v1.pkl'))

logging.basicConfig(filename=os.path.join(
    os.path.abspath(os.path.dirname(__file__)), "simple_example.log"), level=logging.INFO)


def handler(exctype, value, tb):
    logging.error('My Error Information')
    logging.error('Type: {}'.format(str(exctype)))
    logging.error('Value: {}'.format(str(value)))
    logging.error('Traceback: {}'.format(str(traceback.format_tb(tb))))

sys.excepthook = handler


def get_match_data():

    res = requests.get('https://bwf.pass-consulting.com/api/Tournament/44A9F3CE-A822-49C8-A58A-BF165D73CB64/Draw/1/Match?token=7cb1c07a4dd1d13c9545629f51a377fd')
    res = res.json()
    results = res['Result']['ResponseTournamentMatch']
    results = json.dumps(results)
    df = pd.read_json(results)
    df = df[['idTournamentMatch', 'matchTime', 'team1Player1MemberId', 'team1Player1FirstName', 'team1Player1LastName',
             'team2Player1MemberId', 'team2Player1FirstName', 'team2Player1LastName', 'winner']]

    return df


def get_ranking_points():

    res = requests.get('https://bwf.pass-consulting.com/api/Ranking/1C2867DA-3B54-4199-A6E0-85532FD17376/Publication/ED101358-C55F-4F90-9738-94CC7B201325/Category/06816F77-A47E-42B8-9837-308E0451A4C3?token=7cb1c07a4dd1d13c9545629f51a377fd')
    res = res.json()
    points = res['Result']['RankingPublicationPoints']
    df = pd.read_json(json.dumps(points))
    df = df[['player1', 'points']]

    return df


def get_players_ids(match_data):
    ids = []
    for index, data in match_data.iterrows():
        member1_id = data['team1Player1MemberId']
        id1 = get_player_ids(member1_id)
        ids.append(id1)

        member2_id = data['team2Player1MemberId']
        id2 = get_player_ids(member2_id)
        ids.append(id2)

    df = pd.read_json(json.dumps(ids))
    df = df.drop_duplicates(subset=['idPlayer', 'memberId'])
    return df


def get_player_ids(member_id):
    url = 'https://bwf.pass-consulting.com/api/Player/{}?token=7cb1c07a4dd1d13c9545629f51a377fd'.format(member_id)
    res = requests.get(url)
    res = res.json()
    res = res['Result']['Player']
    ids = dict(idPlayer=res['idPlayer'], memberId=res['memberId'])
    return ids



def players_stats(match_data):
    stats = []
    for index, data in match_data.iterrows():
        player1_id = data['team1Player1MemberId']
        player1_stat = get_player_career_stats(player1_id)
        stats.append(player1_stat)

        player2_id = data['team2Player1MemberId']
        player2_stat = get_player_career_stats(player2_id)
        stats.append(player2_stat)

    df = pd.read_json(json.dumps(stats))
    df = df[['memberId', 'totalMatch', 'totalWins', 'totalLosses',
             'currentYearMatch', 'currentYearWins', 'currentYearLosses']]

    df = df.drop_duplicates(subset=['memberId'])

    return df


def get_player_career_stats(member_id):
    url = 'https://bwf.pass-consulting.com/api/Player/{}/CareerStats?token=7cb1c07a4dd1d13c9545629f51a377fd'.format(member_id)
    res = requests.get(url)
    res = res.json()
    for stat in res['Result']['PlayerCareerStats']:
        if stat['type'] == 1:
            return stat


def ids_with_points(ranking_points, players_ids):
    df = ranking_points.merge(players_ids, left_on='player1', right_on='idPlayer')[['memberId', 'points']].drop_duplicates(subset=['memberId', 'points'])
    return df


def get_players_data(stats, players_points):
    df = stats.merge(players_points, on='memberId')
    return df


def start_prediction(classifier):

    logging.info("********************************* Running Initial Prediction: {} *****************************************".format(datetime.now()))

    # This data is saved in "/home/ubuntu/seedion/cron_predictor.log"
    print("********** Running Initial Prediction: ", datetime.now(), " **********")

    start_time = time.time()

    match_data = get_match_data()
    ranking_points = get_ranking_points()
    stats = players_stats(match_data)
    players_ids = get_players_ids(match_data)
    players_ids_points = ids_with_points(ranking_points, players_ids)
    players_data = get_players_data(stats, players_ids_points)

    #players_ranking_points = players_to_ranks(ranking_points)

    games_data = []
    for index, game in match_data.iterrows():
        game_data = buildGameData(game, players_data)
        games_data.append(game_data)

    predictions = predict(games_data, players_data, classifier, False)

    end_time = time.time()
    logging.info("Elapsed: {}".format(end_time-start_time))

    output_to_csv(predictions)

    # This data is saved in "/home/ubuntu/seedion/cron_predictor.log"
    print("********** Finished Initial Prediction **********")

    logging.info("********************************* Finished Running Initial Prediction ********************************")


def get_team_features(teamMemberId, date=datetime.now(), days_before=365, data=None, game_id=None):
    index = data.index[data['memberId'] == teamMemberId]
    index = index.item()

    teamTotalWins = data[data['memberId'] == teamMemberId].at[index, 'totalWins']
    teamTotalLoses = data[data['memberId'] == teamMemberId].at[index, 'totalLosses']

    if teamTotalWins == 0 and teamTotalLoses == 0:
        teamTotalWinsThisSeason = 0
        teamTotalLosesThisSeason = 0
        teamTournamentsPlayedThisSeason = 0
        teamTournamentsPlayed = 0
    else:

        teamTotalWinsThisSeason = data[
            data['memberId'] == teamMemberId].at[index, 'currentYearWins']
        teamTotalLosesThisSeason = data[
            data['memberId'] == teamMemberId].at[index, 'currentYearLosses']
        teamTournamentsPlayedThisSeason = data[
            data['memberId'] == teamMemberId].at[index, 'currentYearMatch']
        teamTournamentsPlayed = data[
            data['memberId'] == teamMemberId].at[index, 'totalMatch']

    averageRankingPoints = data[
        data['memberId'] == teamMemberId].at[index, 'points']

    #print('\nmember ID:', teamMemberId)

    #print('\nteamTotalWins: ', teamTotalWins)
    #print('teamTotalLoses: ', teamTotalLoses)
    #print('teamTotalWinsThisSeason: ', teamTotalWinsThisSeason)
    #print('teamTotalLosesThisSeason: ', teamTotalLosesThisSeason)
    #print('teamTournamentsPlayedThisSeason: ', teamTournamentsPlayedThisSeason)
    #print('teamTournamentsPlayed: ', teamTournamentsPlayed)
    #print('averageRankingPoints: ', averageRankingPoints)
    #print('========================================================================================================')

    return np.nan_to_num(np.asarray(
        [teamTotalWins, teamTotalLoses, teamTotalWinsThisSeason,
        teamTotalLosesThisSeason, teamTournamentsPlayed, averageRankingPoints]
    ))


def predict(games_data, teams_data, clf, menu_mode=True):
    features = {}
    res = []
    pc = 0
    for index, game_features in enumerate(games_data):
        #print(game_features)

        if game_features["teamA"] in features.keys():
            teamA_features = features[game_features["teamA"]]
        else:
            teamA_features = get_team_features(
                game_features["teamAMemberId"], None, None, teams_data
            )
            features[game_features["teamA"]] = teamA_features

        if game_features["teamB"] in features.keys():
            teamB_features = features[game_features["teamB"]]
        else:
            teamB_features = get_team_features(
                game_features["teamBMemberId"], None, None, teams_data
            )
            features[game_features["teamB"]] = teamB_features

        pred_vector = (teamA_features - teamB_features).reshape(1, -1)

        if not pred_vector[0].any():
            print("Skipping zero features data...")
            continue

        pred = clf.predict(pred_vector)
        pred_prob = np.max(clf.predict_proba(pred_vector))

        #print(pred)
        #print(pred_prob)
        if pred[0] == 1:
            winner = game_features["teamA"]
        else:
            winner = game_features["teamB"]

        if datetime.strptime(game_features["date"], '%Y-%m-%d %H:%M:%S') >= datetime.now():
            game_features["winner"] = "Not played yet"
            pred_correct = "Not played yet"
            pc = 0
        else:
            pred_correct = (game_features["winner"] == winner)
            if pred_correct:
                pc += 1

        team_a_total_wins, team_a_total_loses, team_a_total_wins_this_season, team_a_total_loses_this_season, team_a_tournaments_played, team_a_average_ranking_points = teamA_features
        team_b_total_wins, team_b_total_loses, team_b_total_wins_this_season, team_b_total_loses_this_season, team_b_tournaments_played, team_b_average_ranking_points = teamB_features

        pred_data = {
            "game_id" : game_features["id"],
            "team_a_member_id": game_features["teamAMemberId"],
            'team_b_member_id': game_features["teamBMemberId"],
            "team_a" : game_features["teamA"],
            "team_b" : game_features["teamB"],
            "game_winner" : winner,
            "probability" : pred_prob * 100,
            "actual_winner": game_features["winner"],
            "prediction_correct" : pred_correct,
            "game_date":  str(game_features["date"]).replace(" ","T"),
            "tournament_name": game_features["tournamentName"],
            "team_a_players_ranks" : game_features.get('teamA_players_ranks'),
            "team_a_ranking_points": game_features.get('teamA_ranking_points'),
            'team_a_total_wins': team_a_total_wins,
            'team_a_total_loses': team_a_total_loses,
            'team_a_total_wins_this_season': team_a_total_wins_this_season,
            'team_a_total_loses_this_season': team_a_total_loses_this_season,
            'team_a_tournaments_played': team_a_tournaments_played,
            'team_a_average_ranking_points': team_a_average_ranking_points,
            "team_b_players_ranks" : game_features.get('teamB_players_ranks'),
            "team_b_ranking_points": game_features.get('teamB_ranking_points'),
            'team_b_total_wins': team_b_total_wins,
            'team_b_total_loses': team_b_total_loses,
            'team_b_total_wins_this_season': team_b_total_wins_this_season,
            "team_b_total_loses_this_season": team_b_total_loses_this_season,
            'team_b_tournaments_played': team_b_tournaments_played,
            'team_b_average_ranking_points': team_b_average_ranking_points
        }

        #if not menu_mode:
            #print(pred_data)
            #saveToElastic(pred_data)

        res.append(pred_data)

    def print_dict(dictionary):

        print("tournament_name: ", dictionary["tournament_name"])
        print("team_a : ", dictionary["team_a"])
        print("team_b : ", dictionary["team_b"])
        print("predicted game_winner : ", dictionary["game_winner"])
        print("probability : ", dictionary["probability"])
        if "actual_winner" in dictionary.keys():
            print("actual_winner : ", dictionary["actual_winner"])
            print("prediction_correct : ", dictionary.get("prediction_correct", "Not played yet"))
        print("game_date : ", dictionary["game_date"])
        print("game_id : ", dictionary["game_id"])

        #for key, value in sorted(dictionary.items(), key=lambda x: x[0]):
        #    print("{} : {}".format(key, value))

        print("\n")

    #[print_dict(result) for result in res]

    #if menu_mode:
        #print("\nDo you want to save the result into ElasticSearch? (y/n)")
        #og = input(" >>  ")
        #if og.lower().startswith('y'):
            #for pred_data in res:
                #saveToElastic(pred_data)

    return res


def buildGameData(match_data, players_data, team_members=None):
    game_data = {}
    game_data["teamA"] = ' '.join([match_data['team1Player1FirstName'],
                                  match_data['team1Player1LastName']]).lower()
    game_data["teamB"] = ' '.join([match_data['team2Player1FirstName'],
                                  match_data['team2Player1LastName']]).lower()
    game_data["teamAMemberId"] = match_data['team1Player1MemberId']
    game_data['teamBMemberId'] = match_data['team2Player1MemberId']
    game_data['id'] = match_data['idTournamentMatch']
    game_data["date"] = match_data['matchTime']
    game_data["tournamentName"] = 'YONEX All England Open 2020'

    if match_data["winner"] == 1:
        game_data["winner"] = game_data["teamA"]
    else:
        game_data["winner"] = game_data["teamB"]

    player1_points = players_data[
        match_data['team1Player1MemberId'] == players_data['memberId']][['points']]
    game_data['teamA_players_ranks'] = [{
        game_data['teamA']: get_player_points(player1_points)
    }]
    game_data['teamA_ranking_points'] = get_player_points(player1_points)

    player2_points = players_data[
        match_data['team2Player1MemberId'] == players_data['memberId']][['points']]
    game_data['teamB_players_ranks'] = [{
        game_data['teamB']: get_player_points(player2_points)
    }]
    game_data['teamB_ranking_points'] = get_player_points(player2_points)

    return game_data


def get_player_points(points):
    for index, row in points.iterrows():
        return row['points']


def players_to_ranks(palyers_data):
    players_to_ranks = {}
    for index, row in palyers_data.iterrows():
        name = row['Player1Name'].lower()
        points = row['points']
        players_to_ranks[name] = points

    return players_to_ranks


def get_players_names_rankings(team_data):
    players_names_ranks = []
    for index, row in team_data.iterrows():
        player_name = ' '.join([row['PlayerFirstName'], row['PlayerLastName']])
        player_ranking_points = row['PlayerRankingPoints']
        players_names_ranks.append({player_name: player_ranking_points})

    return players_names_ranks


def get_team_ranking_points(team_data):
    return team_data['PlayerRankingPoints'].sum()


def output_to_csv(data):
    with open('prediction_data.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow([
            "game_id", "team_a_member_id", 'team_b_member_id',
            "team_a", "team_b", "game_winner",
            "probability", "actual_winner", 'prediction_correct',
            "game_date", "tournament_name", "team_a_players_ranks",
            "team_a_ranking_points", 'team_a_total_wins',
            'team_a_total_loses', 'team_a_total_wins_this_season',
            'team_a_total_loses_this_season', 'team_a_tournaments_played',
            'team_a_average_ranking_points', "team_b_players_ranks",
            "team_b_ranking_points", 'team_b_total_wins',
            'team_b_total_loses', 'team_b_total_wins_this_season',
            "team_b_total_loses_this_season", 'team_b_tournaments_played',
            'team_b_average_ranking_points'
        ])
        for row in data:
            writer.writerow([
                row["game_id"], row['team_a_member_id'], row['team_b_member_id'],
                row["team_a"], row["team_b"], row["game_winner"], row["probability"],
                row["actual_winner"], row['prediction_correct'],
                row["game_date"], row["tournament_name"], row["team_a_players_ranks"],
                row["team_a_ranking_points"], row['team_a_total_wins'],
                row['team_a_total_loses'], row['team_a_total_wins_this_season'],
                row['team_a_total_loses_this_season'], row['team_a_tournaments_played'],
                row['team_a_average_ranking_points'], row["team_b_players_ranks"],
                row["team_b_ranking_points"], row['team_b_total_wins'],
                row['team_b_total_loses'], row['team_b_total_wins_this_season'],
                row["team_b_total_loses_this_season"], row['team_b_tournaments_played'],
                row['team_b_average_ranking_points']
            ])


if __name__ == '__main__':
    #match_results = get_match_data()
    #print(match_results)
    #players_ranking_points = get_ranking_points()
    #print(players_ranking_points)

    start_prediction(_XGB_CLASSIFIER)

