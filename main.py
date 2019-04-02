# Import the modules needed to run the script.
import sys, os
from datetime import datetime
import redis
from learner_v4 import *
from sklearn.externals import joblib
from sklearn.svm import SVC
import pandas as pd
import xgboost as xgb
import requests
import json
from utils import ELASTIC_CLOUD_PWD, ELASTIC_CLOUD_URL, ELASTIC_CLOUD_USER


pd.set_option('display.height', 100)
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 1000)


# Main definition - constants
menu_actions  = {}

# =======================
#     MENUS FUNCTIONS
# =======================

date = datetime.now()
data = None
clf = None
teamA = ''
teamB = ''
page_size = 20
days_before = 365

# Main menu
def main_menu():
    global data
    global clf
    #os.system('clear')

    print("Welcome,\n")
    #print("Please enter your passcode: ")
    #password = input(" >>  ")
    #if password != "Seedion2018!":
    #    print("You've entered the wrong password. Bye!")
    #    exit()

    print("Choose the following: ")
    print("1. Load features, train the model and predict - this may take long time")
    print("2. Just predict using pretrained model")
    print("\nq. Quit")
    choice = input(" >>  ")
    if choice.lower() == 'q':
        exit()
    elif choice.lower() == '1':
        date_menu()
        days_before_menu()
        print("Getting data from the database and preparing features for training...")
        prepare_features(date, days_before)
        print("Loading features...")
        X, y = load_features(date)
        clf = get_classifier("XGB")
        print("Training...")
        clf.fit(X, y)
    elif choice.lower() == '2':
        clf = joblib.load('XGB_v1.pkl')
        date_menu()
        days_before_menu()

    print("\nLoading data...")
    data = get_future_data(date)
    data['EventStartDate'] = pd.to_datetime(data['EventStartDate'])
    print("\nGames which are going to take place later than ", date, " :\n")

    print(data)

    games_data = predict_menu()
    print("\nExtracting features and running predictions...")
    predict(games_data, clf)

    print("\nWould you like to start over again? (y/n)")
    og = input(" >>  ")
    if og.lower().startswith('y'):
        main_menu()

    return

def predict(games_data, clf, menu_mode=True):
    features = {}
    res = []
    pc = 0
    for game_features in games_data:
        #print(game_features)

        if game_features["teamA"] in features.keys():
            teamA_features = features[game_features["teamA"]]
        else:
            teamA_features = get_team_features(game_features["teamA"], date, days_before, None, game_features["id"])
            features[game_features["teamA"]] = teamA_features

        if game_features["teamB"] in features.keys():
            teamB_features = features[game_features["teamB"]]
        else:
            teamB_features = get_team_features(game_features["teamB"], date, days_before, None, game_features["id"])
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

        if game_features["date"] >= datetime.now():
            game_features["winner"] = "Not played yet"
            pred_correct = "Not played yet"
            pc = 0
        else:
            pred_correct = (game_features["winner"] == winner)
            if pred_correct:
                pc += 1

        pred_data = {
            "game_id" : game_features["id"],
            "team_a" : game_features["teamA"],
            "team_b" : game_features["teamB"],
            "game_winner" : winner,
            "probability" : pred_prob * 100,
            "actual_winner": game_features["winner"],
            #"prediction_correct" : pred_correct,
            "game_date":  str(game_features["date"]).replace(" ","T"),
            "tournament_name": game_features["tournamentName"]
        }

        if game_features["date"] >= datetime.now():
            pc = 0
        else:
            pred_correct = (game_features["winner"] == winner)
            pred_data["prediction_correct"] = pred_correct
            pred_data["actual_winner"] = game_features["winner"]
            if pred_correct:
                pc += 1

        if not menu_mode:
            #print(pred_data)
            saveToElastic(pred_data)

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

    if menu_mode:
        print("\nDo you want to save the result into ElasticSearch? (y/n)")
        og = input(" >>  ")
        if og.lower().startswith('y'):
            for pred_data in res:
                saveToElastic(pred_data)

    return res


def saveToElastic(pred_data):
    print('*************** save to elastic ******************')
    json_data = json.dumps(pred_data)

    test_elastic_url = "https://3289195282f548d8a353ea6edafebd0c.eu-central-1.aws.cloud.es.io:9243/"
    uri = '{}3x3prediction/prediction/{}'.format(test_elastic_url, pred_data["game_id"]) # .format(ELASTIC_CLOUD_URL, pred_data["game_id"])
    response = requests.put(uri, data=json_data, auth=(ELASTIC_CLOUD_USER, ELASTIC_CLOUD_PWD), headers={'content-type': 'application/json'})

    results = json.loads(response.text)

    print(results)

# Date menu
def date_menu():
    print("To start enter a 'cut' date:")
    print("1. I want to enter a date (dd-mm-yyyy)")
    print("2. Use current date")
    print("3. Use 31-12-2016")
    print("\nq. Quit")
    choice = input(" >>  ")
    exec_menu(choice, date_menu_actions)
    print(date)
    return

# Predict Menu
def predict_menu():
    print("Please select how do you want to select teams (by Game Id, Tournamament Name or Team Names directly) for your predictions: \n")

    print("1. Game Id")
    print("2. Tournament Name")
    print("3. Team Names")
    print("4. Select tournament by date")
    print("5. Batch prediction for all tournament games(by start date)")
    print("\nb. Back")
    print("q. Quit")
    choice = input(" >>  ")
    teams = exec_menu(choice, predict_menu_actions)
    return teams

# Timeframe menu
def days_before_menu():
    print("Please select how many days of data before '%s' do you want to extract:" % date)
    print("1. I want to enter a number of days")
    print("2. Use 365 days")
    print("\nq. Quit")
    choice = input(" >>  ")
    exec_menu(choice, days_before_menu_actions)
    return

# Back to main menu
def back():
    menu_actions['main_menu']()

# Exit program
def exit():
    sys.exit()

# =======================
#    MENUS DEFINITIONS
# =======================

def set_date():
    global date
    print("Please enter the date in format dd-mm-yyyy:\n")
    date = input(" >>  ")
    try:
        date = datetime.strptime(date, '%d-%m-%Y')
    except ValueError:
        print("Failed to extrat the date, please try again ...")
        date_menu()
    return

def set_today():
    global date
    date = datetime.now()


def set_default():
    global date
    date = datetime.strptime("31-12-2016", '%d-%m-%Y')

# Execute menu
def exec_menu(choice, menu_actions):
    #os.system('clear')
    ch = choice.lower()
    teams = None
    if ch == '':
        menu_actions['main_menu']()
    else:
        try:
            teams = menu_actions[ch]()
        except (KeyError, Exception) as e:
            print(e)
            print("Invalid selection, please try again.\n")

            menu_actions['main_menu']()
    return teams

# Execute menu
def exec_paged_menu(data, i, columns=None):
    #os.system('clear')
    print("Please select index(he is to the right of the column) which corresponds to GameId/TournamentName/TeamName of the team you would like to choose:\n")

    if len(data) != 0 and columns is not None:
        print(data.iloc[page_size*i:page_size*(i+1)][columns])
    elif len(data) != 0:
        print(data.iloc[page_size*i:page_size*(i+1)])
    else:
        print("No data for selected criteria")
        return None

    if page_size*(i+1) < len(data):
        print("\nn. Next")
    if i > 0:
        print("b. Back")

    print("\nq. Quit")
    choice = input(" >>  ")
    ch = choice.lower()

    if ch == 'n' and page_size*(i+1) < len(data):
        return exec_paged_menu(data, i+1, columns)
    elif ch == 'b' and i > 0:
        return exec_paged_menu(data, i-1, columns)
    elif ch == 'q':
        exit()
    else:
        try:
            key = data.loc[np.int64(ch)]
            game_data = buildGameData(key)

            return [game_data]
        except (KeyError, ValueError) as e:
            print("Invalid selection in menu, please try again.\n")
            print(e)
            return exec_paged_menu(data, i, colunms)

def select_game_id():
    global data

    game = None
    while game is None:
        print("Please enter how your game id should start:\n")
        game_id_beginning = input(" >>  ")
        pages = data[data['GameId'].str.lower().str.lstrip().str.startswith(game_id_beginning.lower().lstrip(), na=False)]
        game = exec_paged_menu(pages, 0)

    return game

def select_tournament_name():
    global data
    game = None
    while game is None:
        print("Please enter how your tournament name should start:\n")
        t_beginning = input(" >>  ")
        pages = data[data['EventName'].str.lower().str.lstrip().str.startswith(t_beginning.lower().lstrip(), na=False)]
        game = exec_paged_menu(pages, 0)

    return game

def select_tournament_by_date():
    global data
    game = None
    while game is None:
        print("Please enter your tournament start date in format dd-mm-yyyy:\n")
        t_date = input(" >>  ")
        tournament_date = datetime.strptime(t_date, '%d-%m-%Y')
        pages = data[data['EventStartDate'].apply(lambda x: x.date()) == tournament_date.date()]
        game = exec_paged_menu(pages, 0)

    return game

def buildGameData(key):
    game_data = {}
    game_data["teamA"] = key['GameTeamNameHome']
    game_data["teamB"] = key['GameTeamNameAway']
    game_data["id"] = key['GameId']
    game_data["date"] = key['EventStartDate']
    game_data["tournamentName"] = key['EventName']
    game_data["winner"] = key["GameTeamNameWinner"]

    return game_data

def select_batch_tournament_by_date():
    global data
    game = None
    while game is None:
        print("Please enter your tournament start date in format dd-mm-yyyy:\n")
        t_date = input(" >>  ")
        tournament_date = datetime.strptime(t_date, '%d-%m-%Y')
        pages = data[data['EventStartDate'].apply(lambda x: x.date()) == tournament_date.date()].drop_duplicates(['EventName'])
        game = exec_paged_menu(pages, 0)

    game = game[0]
    games_date = game['date'].date()

    print("\nSelecting all games from tournament '", game['tournamentName'], "' which starts on ", games_date, " ... \n")
    games = data[(data['EventName'] == game['tournamentName']) & (data['EventStartDate'].apply(lambda x: x.date()) == games_date)].drop_duplicates(['GameId'])
    games_data = []

    print(games)

    for index, game in games.iterrows():
        game_data = buildGameData(game)
        games_data.append(game_data)

    return games_data


def select_teams_name():
    global data
    columns = ['GameTeamNameHome', 'GameTeamNameAway']
    print("Please enter how your team 1 name should start:\n")
    t_beginning = input(" >>  ")
    try:
        pages = data[(data['GameTeamNameHome'].str.lower().str.lstrip().str.startswith(t_beginning.lower().lstrip())) | (data['GameTeamNameAway'].str.lower().str.lstrip().str.startswith(t_beginning.lower().lstrip()))]

        row1 = exec_paged_menu(pages, 0, columns)[0]
        if row1['teamA'].lower().lstrip().startswith(t_beginning.lower().lstrip()) and not row1['teamB'].lower().lstrip().startswith(t_beginning.lower().lstrip()):
            team1 = row1['teamA']
        elif not row1['teamA'].lower().lstrip().startswith(t_beginning.lower().lstrip()) and row1['teamB'].lower().lstrip().startswith(t_beginning.lower().lstrip()):
            team1 = row1['teamB']
        else:
            print("\nPlease select the team you want to choose. It is either Home or Away team from selected row(1 - Home, everything else - Away):\n")
            team_num = input(" >>  ")
            if team_num == '1':
                team1 = row1['teamA']
            else:
                team1 = row1['teamB']

    except Exception as e:
        print(e)

    print("Please enter how your team 2 name should start:\n")
    t_beginning = input(" >>  ")
    try:
        pages = data[(data['GameTeamNameHome'].str.lower().str.lstrip().str.startswith(t_beginning.lower().lstrip())) | (data['GameTeamNameAway'].str.lower().str.lstrip().str.startswith(t_beginning.lower().lstrip()))]

        row1 = exec_paged_menu(pages, 0, columns)[0]
        if row1['teamA'].lower().lstrip().startswith(t_beginning.lower().lstrip()) and not row1['teamB'].lower().lstrip().startswith(t_beginning.lower().lstrip()):
            team2 = row1['teamA']
        elif not row1['teamA'].lower().lstrip().startswith(t_beginning.lower().lstrip()) and row1['teamB'].lower().lstrip().startswith(t_beginning.lower().lstrip()):
            team2 = row1['teamB']
        else:
            print("Please select the team you want to choose. It is either Home or Away team from selected row(1 - Home, everything else - Away):\n")
            team_num = input(" >>  ")
            if team_num == '1':
                team2 = row1['teamA']
            else:
                team2 = row1['teamB']

    except Exception as e:
        print(e)


    game_data = {}
    game_data["teamA"] = team1
    game_data["teamB"] = team2
    game_data["id"] = None
    game_data["date"] = None
    game_data["tournamentName"] = None

    return [game_data]

def set_days_before():
    global days_before
    print("Please enter number of days:\n")
    db = input(" >>  ")
    days_before = int(db)

def set_days_before_default():
    global days_before
    days_before = 365


date_menu_actions = {
    'main_menu': main_menu,
    '1': set_date,
    '2': set_today,
    '3': set_default,
    'q': exit,
}

days_before_menu_actions = {
    'main_menu': main_menu,
    '1': set_days_before,
    '2': set_days_before_default,
    'q': exit,
}

# Menu definition
predict_menu_actions = {
    'main_menu': main_menu,
    '1': select_game_id,
    '2': select_tournament_name,
    '3': select_teams_name,
    '4': select_tournament_by_date,
    '5': select_batch_tournament_by_date,
    'b': back,
    'q': exit,
}

# =======================
#      MAIN PROGRAM
# =======================

# Main Program
if __name__ == "__main__":
    # Launch main menu
    main_menu()
