import pandas as pd
import numpy as np
import os
from nba_api.stats.static import teams
from nba_api.stats.endpoints import leaguegamefinder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime, timedelta

def get_data(filename):
    path = os.getcwd()
    if '.csv' in filename:
        return pd.read_csv(path + filename)
    elif '.pkl' in filename:
        return pd.read_pickle(path + filename)
    else:
        return None

def get_basic_boxscores(date="2018-09-01"):
    nba_teams = teams.get_teams()
    team_names = [team['full_name'] for team in nba_teams]
    team_names.sort()
    team_ids = [team['id'] for team in nba_teams]

    games = None
    for ids in team_ids:
        if games is None:
            gamefinder = leaguegamefinder.LeagueGameFinder(team_id_nullable=ids)
            games = gamefinder.get_data_frames()[0]
        else:
            gamefinder = leaguegamefinder.LeagueGameFinder(team_id_nullable=ids)
            games = pd.concat([games, gamefinder.get_data_frames()[0]])

    games.GAME_ID = pd.to_numeric(games.GAME_ID, downcast='integer')
    games.GAME_DATE = pd.to_datetime(games.GAME_DATE, infer_datetime_format=True)

    today = (datetime.utcnow() - timedelta(hours=9)).strftime('%Y-%m-%d')
    games = games[games['GAME_DATE'] < today]
    games = games[games['GAME_DATE'] > date].sort_values(by='GAME_DATE', ascending=False)

    games.reset_index(drop=True, inplace=True)

    games.sort_values(by='GAME_ID', ascending=False)
    games['HOME_TEAM'] = 0
    games.loc[games['MATCHUP'].str.contains('vs.'), ['HOME_TEAM']] = 1
    games.loc[games['HOME_TEAM'] == 1, :]

    return games

def roll(df, roll_number = 10, procedure = '', suff = '_Roll', selected_columns=[]):
        df_rolling = df[selected_columns + ["TEAM_ABBREVIATION"]]
        df_rolling = df_rolling.groupby(["TEAM_ABBREVIATION"], group_keys=False)

        def find_team_averages(team):
            return team.rolling(roll_number).mean()

        def find_team_medians(team):
            return team.rolling(roll_number).median()

        def find_team_stds(team):
            return team.rolling(roll_number).std()

        if procedure == 'median':
            df_rolling = df_rolling.apply(find_team_medians)
        elif procedure == 'std':
            df_rolling = df_rolling.apply(find_team_stds)
        else:
            procedure = 'mean'
            df_rolling = df_rolling.apply(find_team_averages)

        df_rolling = df_rolling[selected_columns]
        df_rolling = df_rolling.sort_index()

        new_column_names = {}
        for col in df_rolling.columns:
            new_column_names[col] = col + suff + '_' + procedure

        df_rolling = df_rolling.rename(columns=new_column_names)
        return df_rolling

def preprocess_advanced(adv_pickle_filename, roll_methods=['mean'], ohe=True):
    #get basic boxscore data to add columns to the advanced boxscore
    basic = get_data('/backend/data/pkl/raw_games_5yrs.pkl')
    basic = basic.sort_values(by=['GAME_DATE', 'GAME_ID'], ascending=False).reset_index(drop=True)
    games_df = basic[['TEAM_ID', 'TEAM_ABBREVIATION', 'GAME_ID', 'GAME_DATE', 'HOME_TEAM', 'PLUS_MINUS']].copy()

    #get advanced boxscore data from pickle
    advanced = get_data(f'/backend/data/pkl/{adv_pickle_filename}')

    #drop unecessary columns
    columns_to_drop = ['TEAM_CITY', 'MIN', 'E_OFF_RATING', 'E_DEF_RATING',
                   'E_NET_RATING', 'AST_RATIO', 'E_TM_TOV_PCT', 'USG_PCT',
                   'E_USG_PCT', 'E_PACE', 'PACE_PER40', 'PIE']
    advanced = advanced.drop(columns=columns_to_drop)

    #change game_id type to match between the 2 data frames
    advanced['GAME_ID'] = advanced['GAME_ID'].astype('int32')

    #merge the needed columns from basic to advanced
    advanced = advanced.merge(games_df.drop(columns=['TEAM_ID']), on=['GAME_ID', 'TEAM_ABBREVIATION'])

    #drop rows that only have 1 team for the game id
    value_counts = advanced['GAME_ID'].value_counts()
    unique_values = value_counts[value_counts == 1].index.tolist()
    advanced = advanced[~advanced['GAME_ID'].isin(unique_values)].reset_index(drop=True)
    advanced_desc = advanced.sort_values(by=['GAME_DATE'], ascending=True).copy()

    #define features to engineer
    non_eng_features = ['TEAM_ABBREVIATION', 'GAME_ID', 'TEAM_ID', 'TEAM_NAME',
                        'GAME_DATE', 'HOME_TEAM', 'PLUS_MINUS']
    eng_features = advanced_desc.drop(columns=non_eng_features).columns.tolist()

    #caluculate rolling metrics
    if 'mean' in roll_methods:
        df_temp = roll(df = advanced_desc, roll_number=5, procedure='mean', selected_columns=eng_features)
        advanced = advanced.merge(df_temp, left_index=True, right_index=True)
    if 'median' in roll_methods:
        df_temp = roll(df = advanced_desc, roll_number=5, procedure='median', selected_columns=eng_features)
        advanced = advanced.merge(df_temp, left_index=True, right_index=True)
    if 'std' in roll_methods:
        df_temp = roll(df = advanced_desc, roll_number=5, procedure='median', selected_columns=eng_features)
        advanced = advanced.merge(df_temp, left_index=True, right_index=True)

    #split data frame between the home teams and the away teams
    advanced = advanced.sort_values(by=['GAME_DATE', 'GAME_ID', 'HOME_TEAM'], ascending=False).reset_index(drop=True)
    adv_home = advanced.iloc[::2].copy()
    adv_away = advanced.iloc[1::2].copy()

    #change the column names based on home or away
    columns_away = {}
    columns_home = {}
    columns_to_merge = []
    for column in advanced.columns:
        if column == 'GAME_ID' or column == 'PLUS_MINUS' or column == 'GAME_DATE':
            continue
        columns_to_merge.append(column + '_a')
        columns_away[column] = column + '_a'
        columns_home[column] = column + '_h'
    #merge the home and away data frames on to the same game id
    adv_away.rename(columns=columns_away, inplace=True)
    adv_home.rename(columns=columns_home, inplace=True)
    columns_to_merge.append('GAME_ID')
    merged_df = adv_home.merge(adv_away[columns_to_merge], on=['GAME_ID'])
    merged_df = merged_df.dropna()

    #make lists of feature column names
    X_features_num = [col for col in merged_df.columns if 'GAME_ID' not in col
                     and 'GAME_DATE' not in col
                     and 'TEAM_ID' not in col
                     and 'TEAM_NAME' not in col
                     and 'TEAM_ABBREVIATION' not in col
                     and 'PLUS_MINUS' not in col
                     and 'HOME_TEAM' not in col]

    X_features_cat = ['TEAM_ABBREVIATION_h', 'TEAM_ABBREVIATION_a']
    #scale the numerical features
    preproc_data = merged_df.copy()
    scaler = MinMaxScaler()
    preproc_data[X_features_num] = scaler.fit_transform(preproc_data[X_features_num])

    #one hot encode the teams
    if ohe == True:
        ohe = OneHotEncoder(sparse=False)
        ohe.fit(preproc_data[X_features_cat])
        cols = [str(team) +'_h' for team in ohe.categories_[0]] + [str(team) +'_a' for team in ohe.categories_[1]]
        preproc_data[cols]=ohe.transform(preproc_data[X_features_cat])
        X_features = [col for col in preproc_data.columns if 'GAME_ID' not in col
                     and 'GAME_DATE' not in col
                     and 'TEAM_ID' not in col
                     and 'TEAM_NAME' not in col
                     and 'TEAM_ABBREVIATION' not in col
                     and 'PLUS_MINUS' not in col
                     and 'HOME_TEAM' not in col]
    else:
        X_features = [col for col in preproc_data.columns if 'GAME_ID' not in col
                     and 'GAME_DATE' not in col
                     and 'TEAM_ID' not in col
                     and 'TEAM_NAME' not in col
                     and 'PLUS_MINUS' not in col
                     and 'HOME_TEAM' not in col]

    #define and return X and y
    X_preproc = preproc_data[X_features]
    y = preproc_data['PLUS_MINUS']

    return X_preproc, y

# if __name__ == '__main__':
#     X_preproc1, y1 = preprocess_advanced('boxscores_advanced_team_part1.pkl', roll_methods=['mean', 'median', 'std'], ohe=False)
#     X_preproc2, y2 = preprocess_advanced('boxscores_advanced_team_part2.pkl', roll_methods=['mean', 'median', 'std'], ohe=False)
#     X_preproc = pd.concat([X_preproc1, X_preproc2]).reset_index(drop=True)
#     y = pd.concat([y1, y2]).reset_index(drop=True)
#     X_preproc.to_pickle('X_ADV_XGBOOST_TEAM_ALL')
#     y.to_pickle('y_ADV_XGBOOST_TEAM_ALL')
