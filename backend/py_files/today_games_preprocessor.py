import pandas as pd
import numpy as np
import os
from nba_api.stats.static import teams
from nba_api.stats.endpoints import scoreboard
from nba_api.stats.endpoints import leaguegamefinder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime, timedelta

# def get_data(filename):
#     path = os.getcwd()
#     if '.csv' in filename:
#         return pd.read_csv(path + filename)
#     elif '.pkl' in filename:
#         return pd.read_pickle(path + filename)
#     else:
#         return None

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
    games.sort_values(by=['GAME_DATE', 'GAME_ID'], ascending=False).reset_index(drop=True)
    return games

def roll(df, roll_number = 10, procedure = '', suff = '_Roll', selected_columns=[]):
    df_rolling = df[selected_columns + ["TEAM_ABBREVIATION"]]
    df_rolling = df_rolling.groupby(["TEAM_ABBREVIATION"], group_keys=False)

    def find_team_averages(team):
        return team.rolling(roll_number, closed='left').mean()

    def find_team_medians(team):
        return team.rolling(roll_number, closed='left').median()

    def find_team_stds(team):
        return team.rolling(roll_number, closed='left').std()

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

def preprocess_advanced(adv_pickle_filename, roll_methods=['mean'], ohe=True, scaled=True):
    #get basic boxscore data to add columns to the advanced boxscore
    date = datetime.now() - timedelta(days=60)
    date_str = date.strftime('%Y-%m-%d')

    basic = get_basic_boxscores(date=date_str)
    games_df = basic[['TEAM_ID', 'TEAM_ABBREVIATION', 'GAME_ID', 'GAME_DATE', 'HOME_TEAM', 'PLUS_MINUS']].copy()

    #get advanced boxscore data from pickle
    advanced = pd.read_pickle(f'data/pkl/{adv_pickle_filename}')
    # advanced = pd.read_pickle('data/pkl/boxscores_advanced_team_all.pkl')

    ############################################################################
    # Get today's date
    today = datetime.now().strftime('%Y-%m-%d')

    # Get scoreboard for today's games
    scoreboard_today = scoreboard.Scoreboard(game_date=today)
    games = scoreboard_today.game_header.get_data_frame()

    # Get all NBA teams
    nba_teams = teams.get_teams()

    # Create an empty list to store the team data
    team_data = []

    # Loop through each game and add team data to the list
    for index, game in games.iterrows():
        home_team_id = game["HOME_TEAM_ID"]
        away_team_id = game["VISITOR_TEAM_ID"]

        home_team = next((team for team in nba_teams if team["id"] == home_team_id), None)
        away_team = next((team for team in nba_teams if team["id"] == away_team_id), None)

        if home_team is not None and away_team is not None:
            team_data.append({
                "game_id": game["GAME_ID"],
                "home_team_id": home_team["id"],
                "home_team": home_team["abbreviation"],
                "home_team_name": home_team["nickname"],
                "away_team_id": away_team["id"],
                "away_team": away_team["abbreviation"],
                "away_team_name": away_team["nickname"]
            })

    # Convert the list of team data to a DataFrame
    team_df = pd.DataFrame(team_data)

    df1 = team_df[['home_team_id', 'home_team', 'game_id', 'home_team_name']]
    df1.rename(columns={'game_id': 'GAME_ID', 'home_team': 'TEAM_ABBREVIATION', 'home_team_id': 'TEAM_ID', 'home_team_name': 'TEAM_NAME'}, inplace=True)
    df1['GAME_DATE'] = today
    df1['HOME_TEAM'] = 1
    df1['PLUS_MINUS'] = 0
    df2 = team_df[['away_team_id', 'away_team', 'game_id', 'away_team_name']]
    df2.rename(columns={'game_id': 'GAME_ID', 'away_team': 'TEAM_ABBREVIATION', 'away_team_id': 'TEAM_ID', 'away_team_name': 'TEAM_NAME'}, inplace=True)
    df2['GAME_DATE'] = today
    df2['HOME_TEAM'] = 0
    df2['PLUS_MINUS'] = 0
    games_today_df = pd.concat([df1, df2], ignore_index=True, sort=False)
    games_today_df.GAME_DATE = pd.to_datetime(games_today_df.GAME_DATE)

    advanced_today_df = games_today_df.copy()

    columns = ['TEAM_CITY', 'OFF_RATING', 'DEF_RATING',
    'NET_RATING', 'AST_PCT', 'AST_TOV',
    'OREB_PCT', 'DREB_PCT', 'REB_PCT', 'TM_TOV_PCT',
    'EFG_PCT', 'TS_PCT', 'PACE',
    'POSS']

    for column in columns:
        advanced_today_df[column] = 0

    games_today_df.drop(columns=['TEAM_NAME'], inplace=True)

    games_df = pd.concat([games_today_df, games_df], ignore_index=True, sort=False)

    advanced_today_df = advanced_today_df.reindex(columns=['GAME_ID', 'TEAM_ID', 'TEAM_NAME', 'TEAM_ABBREVIATION', 'TEAM_CITY',
                                                    'OFF_RATING', 'DEF_RATING', 'NET_RATING', 'AST_PCT', 'AST_TOV',
                                                    'OREB_PCT', 'DREB_PCT', 'REB_PCT', 'TM_TOV_PCT',
                                                    'EFG_PCT', 'TS_PCT', 'PACE', 'POSS', 'GAME_DATE', 'HOME_TEAM', 'PLUS_MINUS'])

    advanced = pd.concat([advanced_today_df, advanced], ignore_index=True, sort=False)
    ############################################################################

    # games_df = pd.concat([games_today_df, games_df], ignore_index=True, sort=False)

    # advanced_today_df = games_today_df

    # columns = ['TEAM_NAME', 'OFF_RATING', 'DEF_RATING',
    # 'NET_RATING', 'AST_PCT', 'AST_TOV',
    # 'OREB_PCT', 'DREB_PCT', 'REB_PCT', 'TM_TOV_PCT',
    # 'EFG_PCT', 'TS_PCT', 'PACE',
    # 'POSS']

    # for column in columns:
    #     advanced_today_df[column] = 0

    # advanced_today_df = advanced_today_df.reindex(columns=['GAME_ID', 'TEAM_ID', 'TEAM_NAME', 'TEAM_ABBREVIATION',
    #                                                     'OFF_RATING', 'DEF_RATING',
    #                                                     'NET_RATING', 'AST_PCT', 'AST_TOV',
    #                                                     'OREB_PCT', 'DREB_PCT', 'REB_PCT', 'TM_TOV_PCT',
    #                                                     'EFG_PCT', 'TS_PCT', 'PACE', 'POSS'])

    # advanced = pd.concat([advanced_today_df, advanced], ignore_index=True, sort=False)
    ############################################################################

    #change game_id type to match between the 2 data frames
    games_df['GAME_ID'] = games_df['GAME_ID'].astype('int32')
    advanced['GAME_ID'] = advanced['GAME_ID'].astype('int32')

    #merge the needed columns from basic to advanced
    advanced = advanced.merge(games_df.drop(columns=['TEAM_ID', 'GAME_DATE', 'HOME_TEAM', 'PLUS_MINUS']), on=['GAME_ID', 'TEAM_ABBREVIATION'])

    advanced = advanced.drop_duplicates()

    #drop rows that only have 1 team for the game id
    value_counts = advanced['GAME_ID'].value_counts()
    unique_values = value_counts[value_counts == 1].index.tolist()
    advanced = advanced[~advanced['GAME_ID'].isin(unique_values)]
    advanced = advanced.reset_index(drop=True)

    advanced_desc = advanced.sort_values(by=['GAME_DATE'], ascending=True).copy()

    #define features to engineer
    non_eng_features = ['TEAM_ABBREVIATION', 'TEAM_CITY', 'GAME_ID', 'TEAM_ID', 'TEAM_NAME',
                        'GAME_DATE', 'HOME_TEAM', 'PLUS_MINUS']
    eng_features = advanced_desc.drop(columns=non_eng_features).columns.tolist()

    #caluculate rolling metrics
    if 'mean' in roll_methods:
        df_temp = roll(df = advanced_desc, roll_number=4, procedure='mean', selected_columns=eng_features)
        advanced = advanced.merge(df_temp, left_index=True, right_index=True)
    if 'median' in roll_methods:
        df_temp = roll(df = advanced_desc, roll_number=4, procedure='median', selected_columns=eng_features)
        advanced = advanced.merge(df_temp, left_index=True, right_index=True)
    if 'std' in roll_methods:
        df_temp = roll(df = advanced_desc, roll_number=4, procedure='std', selected_columns=eng_features)
        advanced = advanced.merge(df_temp, left_index=True, right_index=True)

    #drop original columns to prevent data leakage
    drop_columns = ['OFF_RATING', 'DEF_RATING', 'NET_RATING', 'AST_PCT', 'AST_TOV', 'OREB_PCT', 'DREB_PCT',
        'REB_PCT', 'TM_TOV_PCT', 'EFG_PCT', 'TS_PCT', 'PACE', 'POSS']
    advanced.drop(columns=drop_columns, inplace=True)

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

    #get elo and raptor scores
    elo_past = pd.read_pickle('data/pkl/elo_past.pkl')


    merged_df = merged_df.merge(elo_past, left_on=['GAME_DATE', 'TEAM_ABBREVIATION_h'], right_on=['date', 'team1'])
    merged_df.drop(columns=['date', 'team1', 'team2'], inplace=True)
    merged_df.rename(columns={'elo1_pre': 'elo_h', 'elo2_pre': 'elo_a', 'raptor1_pre': 'raptor_h', 'raptor2_pre': 'raptor_a'},
          inplace=True)

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
    if scaled == True:
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
    # X_preproc = preproc_data[X_features]
    # y = preproc_data['PLUS_MINUS']
    #print(X_features)

    preproc_data = preproc_data[preproc_data['GAME_DATE'] == datetime.now().strftime('%Y-%m-%d')]
    return preproc_data , X_features

if __name__ == '__main__':
    today_date = datetime.today().strftime('%Y-%m-%d')
    preproc_part_today, X_features = preprocess_advanced('boxscores_advanced_team_all.pkl',
                                        roll_methods=['mean', 'median', 'std'],
                                        ohe=True,
                                        scaled=False)
    preproc_part_today.to_pickle(f'data/pkl/demo_{today_date}.pkl')

#     preproc_part2, X_features = preprocess_advanced('boxscores_advanced_team_part2.pkl',
#                                         roll_methods=['mean'],
#                                         ohe=True,
#                                         scaled=False)
#     preproc_all = pd.concat([preproc_part1, preproc_part2]).reset_index(drop=True)
#     preproc_all.to_pickle('/data/pkl/alec_test_data.pkl')
