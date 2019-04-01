import sys
import os
from datetime import datetime, timedelta
import time
import logging
import traceback

from sklearn.externals import joblib

from learner_v4 import get_future_data
from main import buildGameData, predict


_DAYS_IN_FUTURE = 92
xgb_classifier = joblib.load(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'XGB_v1.pkl'))
logging.basicConfig(filename=os.path.join(os.path.abspath(os.path.dirname(__file__)), "predictor.log"), level=logging.INFO)


def handler(exctype, value, tb):
    logging.error('My Error Information')
    logging.error('Type: {}'.format(str(exctype)))
    logging.error('Value: {}'.format(str(value)))
    logging.error('Traceback: {}'.format(str(traceback.format_tb(tb))))

sys.excepthook = handler


def start_prediction(num_days, classifier):

    logging.info("********************************* Running Initial Prediction: {} *****************************************".format(datetime.now()))

    # This data is saved in "/home/ubuntu/seedion/cron_predictor.log"
    print("********** Running Initial Prediction: ", datetime.now(), " **********")

    date = datetime.now()
    future_date = date + timedelta(days=num_days)

    start_time = time.time()
    data = get_future_data(date, future_date)

    games_data = []
    for index, game in data.iterrows():
        game_data = buildGameData(game)
        games_data.append(game_data)

    predictions = predict(games_data, classifier, False)

    end_time = time.time()
    logging.info("Elapsed: {}".format(end_time-start_time))

    # This data is saved in "/home/ubuntu/seedion/cron_predictor.log"
    print("********** Finished Initial Prediction **********")

    logging.info("********************************* Finished Running Initial Prediction ********************************")


if __name__ == '__main__':
    start_prediction(_DAYS_IN_FUTURE, xgb_classifier)
