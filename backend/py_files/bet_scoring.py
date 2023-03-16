import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.metrics import accuracy_score

def get_historical_preds():
    y_pred = pd.read_pickle('../y_pred_70.pkl')
    y_pred['left_0.7'] = round(y_pred['left_0.7'], 1)
    y_pred['right_0.7'] = round(y_pred['right_0.7'], 1)
    return y_pred

def get_today_preds():
    y_pred = pd.read_pickle('../y_pred_today.pkl')
    y_pred['left_0.7'] = round(y_pred['left_0.7'], 1)
    y_pred['right_0.7'] = round(y_pred['right_0.7'], 1)
    return y_pred

def get_historical_betting_data():
    bets = pd.read_pickle('../data/pkl/sbr_betting_data.pkl')
    bets = bets.reset_index(drop=True)
    bets['Game_Date'] = pd.to_datetime(bets['Game_Date'])
    sportsbooks_spreads = ['Opening_Spread','Betmgm_Spread', 'Draft_Kings_Spread', 'Fanduel_Spread',
                       'Caesars_Spread', 'Pointsbet_Spread', 'Wynn_Spread', 'Betrivers_Spread']

    sportsbooks_odds = ['Opening_Odds', 'Betmgm_Odds', 'Draft_Kings_Odds', 'Fanduel_Odds',
                        'Caesars_Odds', 'Pointsbet_Odds', 'Wynn_Odds', 'Betrivers_Odds']

    sportsbooks_spreads_odds = ['Betmgm_Spread', 'Betmgm_Odds',
           'Draft_Kings_Spread', 'Draft_Kings_Odds', 'Fanduel_Spread',
           'Fanduel_Odds', 'Caesars_Spread', 'Caesars_Odds', 'Pointsbet_Spread',
           'Pointsbet_Odds', 'Wynn_Spread', 'Wynn_Odds', 'Betrivers_Spread',
           'Betrivers_Odds']

    bets = bets.replace(to_replace=['-', ''], value='1000')
    bets = bets.replace(to_replace=[1000, '1000'], value=[np.nan, ''])
    bets = bets.replace(to_replace=['PK'], value='0')

    bets[sportsbooks_spreads] = bets[sportsbooks_spreads].astype('float32')
    bets[sportsbooks_odds] = bets[sportsbooks_odds].astype('int32')
    bets['Game_Date'] = pd.to_datetime(bets['Game_Date'])
    return bets

def get_today_betting_data():
    bets = pd.read_pickle('../data/pkl/sbr_today_betting_data.pkl')
    bets = bets.reset_index(drop=True)
    bets['Game_Date'] = pd.to_datetime(bets['Game_Date'])
    sportsbooks_spreads = ['Opening_Spread','Betmgm_Spread', 'Draft_Kings_Spread', 'Fanduel_Spread',
                       'Caesars_Spread', 'Pointsbet_Spread', 'Wynn_Spread', 'Betrivers_Spread']

    sportsbooks_odds = ['Opening_Odds', 'Betmgm_Odds', 'Draft_Kings_Odds', 'Fanduel_Odds',
                        'Caesars_Odds', 'Pointsbet_Odds', 'Wynn_Odds', 'Betrivers_Odds']

    sportsbooks_spreads_odds = ['Betmgm_Spread', 'Betmgm_Odds',
           'Draft_Kings_Spread', 'Draft_Kings_Odds', 'Fanduel_Spread',
           'Fanduel_Odds', 'Caesars_Spread', 'Caesars_Odds', 'Pointsbet_Spread',
           'Pointsbet_Odds', 'Wynn_Spread', 'Wynn_Odds', 'Betrivers_Spread',
           'Betrivers_Odds']

    bets = bets.replace(to_replace=['-', ''], value='1000')
    bets = bets.replace(to_replace=[1000, '1000'], value=[np.nan, ''])
    bets = bets.replace(to_replace=['PK'], value='0')

    bets[sportsbooks_spreads] = bets[sportsbooks_spreads].astype('float32')
    bets[sportsbooks_odds] = bets[sportsbooks_odds].astype('int32')
    bets['Game_Date'] = pd.to_datetime(bets['Game_Date'])
    return bets


def merge_today_predictions_with_bet_data(y_pred, bets):
    y_pred['TEAM_NAME_a'].sort_values().unique()
    team_name = ['Hawks', 'Celtics', 'Nets', 'Hornets', 'Bulls', 'Cavaliers', 'Mavericks', 'Nuggets', 'Pistons',
                'Warriors', 'Rockets', 'Pacers', 'Clippers', 'Lakers', 'Grizzlies', 'Heat', 'Bucks', 'Timberwolves',
                'Pelicans', 'Knicks', 'Thunder', 'Magic', '76ers', 'Suns', 'Trail Blazers', 'Kings', 'Spurs',
                'Raptors', 'Jazz', 'Wizards']
    team_city = bets['Team_Name'].sort_values().unique().tolist()
    map = {}
    for i, name in enumerate(team_name):
        map[name] = team_city[i]
    y_pred = y_pred.replace({'TEAM_NAME_h': map, 'TEAM_NAME_a': map})

    home=[]
    for i in range(len(bets)):
        row=bets.iloc[i]
        home.append([row.Opponent if row.Home==0 else row.Team_Name][0])
    bets['TEAM_NAME_h'] = home

    compare = bets.merge(y_pred, left_on=['Game_Date', 'TEAM_NAME_h'], right_on=['GAME_DATE', 'TEAM_NAME_h'])
    compare = compare[['Game_Date', 'GAME_ID', 'Team_Name', 'Home', 'Opponent','Opening_Spread', 'left_0.7', 'right_0.7']]
    compare['y_pred'] = np.where((compare['Home'] == 1), compare['right_0.7'], compare['left_0.7'])
    compare.loc[compare['Home'] == 1, ['y_pred']] = compare['y_pred'] * -1
    #compare.loc[compare['Home'] == 0, ['PLUS_MINUS']] = compare['PLUS_MINUS'] * -1
    compare = compare.drop(columns=['left_0.7', 'right_0.7'])
    return compare


def merge_historical_predictions_with_bet_data(y_pred, bets):
    y_pred['TEAM_NAME_a'].sort_values().unique()
    team_name = ['Hawks', 'Celtics', 'Nets', 'Hornets', 'Bulls', 'Cavaliers', 'Mavericks', 'Nuggets', 'Pistons',
                'Warriors', 'Rockets', 'Pacers', 'Clippers', 'Lakers', 'Grizzlies', 'Heat', 'Bucks', 'Timberwolves',
                'Pelicans', 'Knicks', 'Thunder', 'Magic', '76ers', 'Suns', 'Trail Blazers', 'Kings', 'Spurs',
                'Raptors', 'Jazz', 'Wizards']
    team_city = bets['Team_Name'].sort_values().unique().tolist()
    map = {}
    for i, name in enumerate(team_name):
        map[name] = team_city[i]
    y_pred = y_pred.replace({'TEAM_NAME_h': map, 'TEAM_NAME_a': map})

    home=[]
    for i in range(len(bets)):
        row=bets.iloc[i]
        home.append([row.Opponent if row.Home==0 else row.Team_Name][0])
    bets['TEAM_NAME_h'] = home

    compare = bets.merge(y_pred, left_on=['Game_Date', 'TEAM_NAME_h'], right_on=['GAME_DATE', 'TEAM_NAME_h'])
    compare = compare[['Game_Date', 'GAME_ID', 'Team_Name', 'Home', 'Opponent','Opening_Spread', 'PLUS_MINUS', 'left_0.7', 'right_0.7']]
    compare['y_pred'] = np.where((compare['Home'] == 1), compare['right_0.7'], compare['left_0.7'])
    compare.loc[compare['Home'] == 1, ['y_pred']] = compare['y_pred'] * -1
    #compare.loc[compare['Home'] == 0, ['PLUS_MINUS']] = compare['PLUS_MINUS'] * -1
    compare = compare.drop(columns=['left_0.7', 'right_0.7'])
    return compare

def predict_today_bets(compare):
    compare['bet'] = np.where((compare['y_pred'] < compare['Opening_Spread']), True, False)
    return compare

def predict_historical_bets(compare):
    compare['bet'] = np.where((compare['y_pred'] < compare['Opening_Spread']), True, False)
    compare['winning_bet'] = np.where(compare['PLUS_MINUS'] > -compare['Opening_Spread'], True, False)
    return compare

def score_bets(compare):
    grouped_game_ids = compare.groupby(by = 'GAME_ID', as_index=False)['bet'].sum()
    scorable_game_ids = grouped_game_ids.loc[grouped_game_ids['bet'] == 1, ['GAME_ID']]
    game_ids = [x.item() for x in scorable_game_ids.values]
    scorable_bets = compare.loc[compare['GAME_ID'].isin(game_ids), :]
    betting_accuracy = accuracy_score(scorable_bets['winning_bet'], scorable_bets['bet'])
    print(betting_accuracy)


if __name__ == "__main__":
    y_pred = get_today_preds()
    bets = get_today_betting_data()
    compare = merge_today_predictions_with_bet_data(y_pred=y_pred, bets=bets)
    predicted_bet = predict_today_bets(compare=compare)
