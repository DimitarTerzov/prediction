import pandas as pd
import requests


def get_data(url):

    response = requests.get(url)
    data = pd.DataFrame(response.json())

    rows = data.drop_duplicates(['GameId'])
    rows.sort_values(by=['EventStartDate'])

    print(rows.shape)

    return rows, data


def main(url):
    print("********** Running Initial Prediction **********")

    data, teams_data = get_data(url)

    #games_data = []
    #for index, game in data.iterrows():
        #game_data = buildGameData(game, teams_data)
        #games_data.append(game_data)

    #predictions = predict(games_data, classifier, False)

    print("********** Finished Initial Prediction **********")


if __name__ == '__main__':
    URL = "https://bwf-stage.pass-consulting.com/api/v1/custom/matches_rankings?limit=10&params=start_date%3D'2016-08-20'%2Cend_date%3D'2016-08-20'&token=7cb1c07a4dd1d13c9545629f51a377fd"
    main(URL)
