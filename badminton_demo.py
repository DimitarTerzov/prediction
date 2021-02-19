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

_XGB_CLASSIFIER = joblib.load(
    os.path.join(os.path.abspath(os.path.dirname(__file__)), 'XGB_v1.pkl'))


def get_games_data(start, end, limit):

    url = 'https://bwf-stage.pass-consulting.com/api/v1/custom/matches_rankings'
    params = {
        'limit': limit,
        'params': "start_date='{}',end_date='{}'".format(start, end),
        'token': '7cb1c07a4dd1d13c9545629f51a377fd'
    }
    res = requests.get(url, params=params)
    res = res.json()
    res = json.dumps(res)
    df = pd.read_json(res)
    df = df[[
        'tournament_name', 'event_name', 'match_time', 'match_winner',
        'match_team1_player1_id', 'match_team1_player1_first_name', 'match_team1_player1_last_name',
        'match_team2_player1_id', 'match_team2_player1_first_name', 'match_team2_player1_last_name',
        'ranking_team1_ranking_points', 'wins_year_team1_player1', 'losses_year_team1_player1',
        'ranking_team2_ranking_points', 'wins_year_team2_player1', 'losses_year_team2_player1',
        'matches_total_team1_player1', 'wins_total_team1_player1', 'losses_total_team1_player1',
        'matches_total_team2_player1', 'wins_total_team2_player1', 'losses_total_team2_player1'
    ]]

    return df


def build_game_data(data, index):
    game_data = {}
    game_data["teamA"] = ' '.join([data['match_team1_player1_first_name'],
                                   data['match_team1_player1_last_name']]).lower()
    game_data["teamB"] = ' '.join([data['match_team2_player1_first_name'],
                                   data['match_team2_player1_last_name']]).lower()
    game_data["teamAMemberId"] = data['match_team1_player1_id']
    game_data['teamBMemberId'] = data['match_team2_player1_id']
    game_data['id'] = index
    game_data["date"] = data['match_time']
    game_data["tournamentName"] = data['tournament_name']

    if data["match_winner"] == 1:
        game_data["winner"] = game_data["teamA"]
    else:
        game_data["winner"] = game_data["teamB"]

    player1_points = data['ranking_team1_ranking_points']
    game_data['teamA_players_ranks'] = [{
        game_data['teamA']: player1_points
    }]
    game_data['teamA_ranking_points'] = player1_points

    player2_points = data['ranking_team2_ranking_points']
    game_data['teamB_players_ranks'] = [{
        game_data['teamB']: player2_points
    }]
    game_data['teamB_ranking_points'] = player2_points

    game_data['teamA_total_wins'] = data['wins_total_team1_player1']
    game_data['teamA_total_losses'] = data['losses_total_team1_player1']
    game_data['teamA_total_wins_this_season'] = data['wins_year_team1_player1']
    game_data['teamA_total_losses_this_season'] = data['losses_year_team1_player1']
    game_data['teamA_total_matches'] = data['matches_total_team1_player1']

    game_data['teamB_total_wins'] = data['wins_total_team2_player1']
    game_data['teamB_total_losses'] = data['losses_total_team2_player1']
    game_data['teamB_total_wins_this_season'] = data['wins_year_team2_player1']
    game_data['teamB_total_losses_this_season'] = data['losses_year_team2_player1']
    game_data['teamB_total_matches'] = data['matches_total_team2_player1']

    return game_data


def get_team_b_features(team_data):

    team_total_wins = team_data['teamB_total_wins']
    team_total_loses = team_data['teamB_total_losses']

    if team_total_wins == 0 and team_total_loses == 0:
        team_total_wins_this_season = 0
        team_total_loses_this_season = 0
        team_tournaments_played = 0

    else:
        team_total_wins_this_season = team_data['teamB_total_wins_this_season']
        team_total_loses_this_season = team_data['teamB_total_losses_this_season']
        team_tournaments_played = team_data['teamB_total_matches']

    average_ranking_points = team_data['teamB_ranking_points']

    return np.nan_to_num(np.asarray(
        [team_total_wins, team_total_loses, team_total_wins_this_season,
         team_total_loses_this_season, team_tournaments_played, average_ranking_points]
    ))


def get_team_a_features(team_data):

    team_total_wins = team_data['teamA_total_wins']
    team_total_loses = team_data['teamA_total_losses']

    if team_total_wins == 0 and team_total_loses == 0:
        team_total_wins_this_season = 0
        team_total_loses_this_season = 0
        team_tournaments_played = 0

    else:
        team_total_wins_this_season = team_data['teamA_total_wins_this_season']
        team_total_loses_this_season = team_data['teamA_total_losses_this_season']
        team_tournaments_played = team_data['teamA_total_matches']

    average_ranking_points = team_data['teamA_ranking_points']

    return np.nan_to_num(np.asarray(
        [team_total_wins, team_total_loses, team_total_wins_this_season,
         team_total_loses_this_season, team_tournaments_played, average_ranking_points]
    ))


def predict(games_data, clf):
    features = {}
    res = []
    pc = 0
    for index, game_features in enumerate(games_data):

        if game_features["teamA"] in features.keys():
            team_a_features = features[game_features["teamA"]]
        else:
            team_a_features = get_team_a_features(game_features)
            features[game_features["teamA"]] = team_a_features

        if game_features["teamB"] in features.keys():
            team_b_features = features[game_features["teamB"]]
        else:
            team_b_features = get_team_b_features(game_features)
            features[game_features["teamB"]] = team_b_features

        pred_vector = (team_a_features - team_b_features).reshape(1, -1)

        if not pred_vector[0].any():
            print("Skipping zero features data...")
            continue

        pred = clf.predict(pred_vector)
        pred_prob = np.max(clf.predict_proba(pred_vector))

        if pred[0] == 1:
            winner = game_features["teamA"]
        else:
            winner = game_features["teamB"]

        if game_features["date"] >= datetime.now():
            game_features["winner"] = "Not played yet"
            pred_correct = "Not played yet"
            pc = 0
        else:
            pred_correct = (game_features["winner"] == winner)
            if pred_correct:
                pc += 1

        (
            team_a_total_wins, team_a_total_loses,
            team_a_total_wins_this_season, team_a_total_loses_this_season,
            team_a_tournaments_played, team_a_average_ranking_points
        ) = team_a_features
        (
            team_b_total_wins, team_b_total_loses,
            team_b_total_wins_this_season, team_b_total_loses_this_season,
            team_b_tournaments_played, team_b_average_ranking_points
        ) = team_b_features

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

        res.append(pred_data)

    return res


def output_to_csv(data):
    with open('prediction_data.csv', 'w', newline='', encoding='utf-8') as csvfile:
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


def get_player_points(points):
    for index, row in points.iterrows():
        return row['ranking_team1_ranking_points']


def get_date(prompt):
    while True:
        value = input(prompt)
        try:
            datetime.strptime(value, "%Y-%m-%d")
        except ValueError:
            print("Wrong date format")
            continue
        break
    return value


def get_limit(prompt):
    while True:
        try:
            value = int(input(prompt))
        except ValueError:
            print("Limit should be a number!")
            continue
        break
    return value


def main():
    start_date = get_date("Please enter `start date` in following format `YYYY-MM-DD`>>>")
    end_date = get_date("Please enter `end date` in following format `YYYY-MM-DD`>>>")
    limit = get_limit("Please enter `limit` to number for the queried games>>>")
    print("Starting prediction...")
    data = get_games_data(start_date, end_date, limit)
    games_data = []
    for index, row in data.iterrows():
        game_data = build_game_data(row, index)
        games_data.append(game_data)

    predictions = predict(games_data, _XGB_CLASSIFIER)
    output_to_csv(predictions)
    print("Prediction Done!")
    print("Please check your results in `prediction_data.csv`")


if __name__ == '__main__':
    main()
