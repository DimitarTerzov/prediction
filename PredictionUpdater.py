# Import the modules needed to run the script.
import traceback
import sys, os
from datetime import datetime, timedelta
import redis
import yaml
from learner_v4 import *
from main import *
from sklearn.externals import joblib
from sklearn.svm import SVC
import pandas as pd
import xgboost as xgb
import requests
import json
import schedule
import logging
from joblib import Parallel, delayed
import multiprocessing
from utils import ELASTIC_CLOUD_USER, ELASTIC_CLOUD_PWD, ELASTIC_CLOUD_URL


clf = joblib.load(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'XGB_v1.pkl'))
logging.basicConfig(filename="predictor.log", level=logging.INFO)
num_cores = multiprocessing.cpu_count()
num_days = 92

def handler(exctype, value, tb):
    logging.error('My Error Information')
    logging.error('Type: {}'.format(str(exctype)))
    logging.error('Value: {}'.format(str(value)))
    logging.error('Traceback: {}'.format(str(traceback.format_tb(tb))))

sys.excepthook = handler

def runStartPrediction():

    logging.info("********************************* Running Initial Prediction *****************************************")

    date = datetime.now()
    future_date = date + timedelta(days=num_days)

    start_time = time.time()
    data = get_future_data(date, future_date)

    games_data = []
    for index, game in data.iterrows():
        game_data = buildGameData(game)
        games_data.append(game_data)

    predictions = predict(games_data, clf, False)

    end_time = time.time()
    logging.info("Elapsed: {}".format(end_time-start_time))

    logging.info("********************************* Finished Running Initial Prediction *****************************************")


def updatePreviousResults():
    today = datetime.now()
    yesterday = (today - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    days_before = (today - timedelta(days=7)).replace(hour=0, minute=0, second=0, microsecond=0)

    logging.info("********************************* Updating Previous Results ('%s') *****************************************" % today)

    data = json.dumps({
        "query": {
            "bool": {
                "must_not": {
                    "exists": {
                        "field": "actual_winner"
                    }
            },
                "filter": {
                    "range": {
                        "game_date": {
                             "gt": str(days_before).replace(" ","T"),
                             "lt": str(today).replace(" ","T")
                        }
                    }
                }
            }
        }
    }, default=str)

    uri = '{}3x3prediction/_search'.format(ELASTIC_CLOUD_URL)
    response = requests.get(uri, data=data, auth=(ELASTIC_CLOUD_USER, ELASTIC_CLOUD_PWD), headers={'content-type': 'application/json'})
    results = json.loads(response.text)

    game_ids_dict = {}
    for result in results["hits"]["hits"]:
        game_ids_dict[result["_source"]["game_id"]] = result["_source"]["game_winner"]

    logging.info("Games to check winner: ")
    logging.info(list(game_ids_dict.keys()))

    data_with_winner = get_winners_by_game_ids(days_before, today, list(game_ids_dict.keys()))

    for index, game_with_winner in data_with_winner.iterrows():
        game_id = game_with_winner["GameId"]
        actual_winner = game_with_winner["GameTeamNameWinner"]

        uri = '{}3x3prediction/prediction/{}{}'.format(ELASTIC_CLOUD_URL, game_id, '/_update')

        prediction_correct = actual_winner == game_ids_dict[game_id]
        logging.info("game_id: {} actual_winner: {} prediction_correct: {}".format(game_id, actual_winner, prediction_correct))

        json_data = json.dumps({
            "script": "ctx._source.actual_winner = '%s'; ctx._source.prediction_correct = '%s';" % (actual_winner.replace("'","\\'").replace('"', '\\"'), str(prediction_correct).lower()),
            "lang": "painless",
        })

        response = requests.post(uri, data=json_data, auth=(ELASTIC_CLOUD_USER, ELASTIC_CLOUD_PWD), headers={'content-type': 'application/json'})

        results = json.loads(response.text)
        logging.info("Elastic update results: ")
        logging.info(results)

    logging.info("********************************* Finished Updating Previous Results *****************************************")


def runEveryDayPredictions():
    today = datetime.now()
    day_before = today + timedelta(days=num_days-1)
    next_day = today + timedelta(days=num_days)

    logging.info("********************************* Running Predictions For ('%s') *****************************************" % today)

    data = get_future_data(day_before, next_day)

    games_data = []
    for index, game in data.iterrows():
        game_data = buildGameData(game)
        games_data.append(game_data)

    predictions = predict(games_data, clf, False)

    logging.info("********************************* Finished Running Predictions *****************************************")

# =======================
#      MAIN PROGRAM
# =======================

# Main Program
if __name__ == "__main__":
    global start_date
    start_date = datetime.now()

    global num_days
    num_days = 92

    #runStartPrediction()

    schedule.every(45).minutes.do(updatePreviousResults)
    schedule.every().day.at("01:30").do(runStartPrediction)

    #runStartPrediction(num_days)

    #schedule.every(1).minutes.do(updatePreviousResults)
    #schedule.every(10).minutes.do(runEveryDayPredictions)

    while True:
        schedule.run_pending()
        time.sleep(500)

