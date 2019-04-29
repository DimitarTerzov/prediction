import sys
import os
from datetime import datetime, timedelta
import traceback
import logging
import requests
import json

from utils import ELASTIC_CLOUD_URL, ELASTIC_CLOUD_USER, ELASTIC_CLOUD_PWD
from learner_v4 import get_winners_by_game_ids


logging.basicConfig(filename=os.path.join(os.path.abspath(os.path.dirname(__file__)), "predictor.log"), level=logging.INFO)


def handler(exctype, value, tb):
    logging.error('My Error Information')
    logging.error('Type: {}'.format(str(exctype)))
    logging.error('Value: {}'.format(str(value)))
    logging.error('Traceback: {}'.format(str(traceback.format_tb(tb))))

sys.excepthook = handler


def update_previous_results():
    today = datetime.now()
    yesterday = (today - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    days_before = (today - timedelta(days=7)).replace(hour=0, minute=0, second=0, microsecond=0)

    logging.info("********************************* Updating Previous Results ('%s') *****************************************" % today)

    # This data is saved in "/home/ubuntu/seedion/cron_updator.log"
    print("***********Update Prections {}******".format(today))

    data = json.dumps({
        "size": 100,
        "query": {
            "bool": {
                "should": [
                    {"match": {"actual_winner.keyword": ""}},
                    {"match": {"actual_winner": "Not played yet"}}
                ],
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

        prediction_correct = actual_winner.lower() == game_ids_dict[game_id].lower()
        logging.info("game_id: {} actual_winner: {} prediction_correct: {}".format(game_id, actual_winner, prediction_correct))

        json_data = json.dumps({
            "script": "ctx._source.actual_winner = '%s'; ctx._source.prediction_correct = '%s';" % (actual_winner.replace("'","\\'"), str(prediction_correct).lower()),
            "lang": "painless",
        })

        response = requests.post(uri, data=json_data, auth=(ELASTIC_CLOUD_USER, ELASTIC_CLOUD_PWD), headers={'content-type': 'application/json'})

        results = json.loads(response.text)
        logging.info("Elastic update results: ")
        logging.info(results)

        # This data is saved in "/home/ubuntu/seedion/cron_updator.log"
        print("******Update Elastic***** \n", results)

    logging.info("********************************* Finished Updating Previous Results *****************************************")


if __name__ == '__main__':
    update_previous_results()
