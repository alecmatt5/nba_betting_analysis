import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

def get_data(filename):
    if '.csv' in filename:
        return pd.read_csv(filename)
    elif '.pkl' in filename:
        return pd.read_pickle(filename)
    else:
        return None

def preprocess_advanced(adv_filename):
    #get basic boxscore data to add columns to the advanced boxscore
    basic = get_data('../data/pkl/raw_games_5yrs.pkl')
    basic = basic.sort_values(by=['GAME_DATE', 'GAME_ID'], ascending=False).reset_index(drop=True)
    games_df = basic[['TEAM_ID', 'TEAM_ABBREVIATION', 'GAME_ID', 'GAME_DATE', 'HOME_TEAM', 'PTS', 'PLUS_MINUS']].copy()

    #get advanced boxscore data from pickle
    advanced = get_data(f'../data/pkl/{adv_filename}')

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

    #caluculate rolling average
    rolling_features = ['OFF_RATING', 'DEF_RATING', 'NET_RATING', 'AST_PCT',
       'AST_TOV', 'OREB_PCT', 'DREB_PCT', 'REB_PCT', 'TM_TOV_PCT', 'EFG_PCT',
       'TS_PCT', 'PACE', 'POSS']
    advanced_desc = advanced_desc.groupby('TEAM_ABBREVIATION', as_index=False, group_keys=False)[rolling_features].rolling(5).mean()
    advanced_desc.drop(columns=['TEAM_ABBREVIATION']).sort_index()
    advanced[rolling_features] = advanced_desc.drop(columns=['TEAM_ABBREVIATION']).sort_index()
    advanced = advanced.sort_values(by=['GAME_DATE', 'GAME_ID', 'HOME_TEAM'], ascending=False).reset_index(drop=True)

    #split data frame between the home teams and the away teams
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
    columns_to_merge.append('GAME_ID')
    adv_away.rename(columns=columns_away, inplace=True)
    adv_home.rename(columns=columns_home, inplace=True)

    #merge the home and away data frames on to the same game id
    merged_df = adv_home.merge(adv_away[columns_to_merge], on=['GAME_ID'])
    merged_df = merged_df.dropna()

    #reorder the columns
    merged_df = merged_df[['GAME_ID', 'GAME_DATE', 'TEAM_ID_h', 'TEAM_NAME_h',
       'TEAM_ABBREVIATION_h', 'HOME_TEAM_h', 'OFF_RATING_h', 'DEF_RATING_h',
       'NET_RATING_h', 'AST_PCT_h', 'AST_TOV_h', 'OREB_PCT_h', 'DREB_PCT_h',
       'REB_PCT_h', 'TM_TOV_PCT_h', 'EFG_PCT_h', 'TS_PCT_h', 'PACE_h',
       'POSS_h', 'PTS_h', 'TEAM_ID_a', 'TEAM_NAME_a',
       'TEAM_ABBREVIATION_a', 'HOME_TEAM_a', 'OFF_RATING_a', 'DEF_RATING_a',
       'NET_RATING_a', 'AST_PCT_a', 'AST_TOV_a', 'OREB_PCT_a', 'DREB_PCT_a',
       'REB_PCT_a', 'TM_TOV_PCT_a', 'EFG_PCT_a', 'TS_PCT_a', 'PACE_a',
       'POSS_a', 'PTS_a', 'PLUS_MINUS']]

    #make lists of feature column names
    X_features_num = ['OFF_RATING_h', 'DEF_RATING_h',
       'NET_RATING_h', 'AST_PCT_h', 'AST_TOV_h', 'OREB_PCT_h', 'DREB_PCT_h',
       'REB_PCT_h', 'TM_TOV_PCT_h', 'EFG_PCT_h', 'TS_PCT_h', 'PACE_h',
       'POSS_h', 'OFF_RATING_a', 'DEF_RATING_a',
       'NET_RATING_a', 'AST_PCT_a', 'AST_TOV_a', 'OREB_PCT_a', 'DREB_PCT_a',
       'REB_PCT_a', 'TM_TOV_PCT_a', 'EFG_PCT_a', 'TS_PCT_a', 'PACE_a',
       'POSS_a']
    X_features_cat = ['TEAM_ABBREVIATION_h', 'TEAM_ABBREVIATION_a']

    #scale the numerical features
    X = merged_df.copy()
    scaler = MinMaxScaler()
    X[X_features_num] = scaler.fit_transform(X[X_features_num])

    #one hot encode the teams
    ohe = OneHotEncoder(sparse=False)
    ohe.fit(X[X_features_cat])
    cols = [str(team) +'_h' for team in ohe.categories_[0]] + [str(team) +'_a' for team in ohe.categories_[1]]
    X[cols]=ohe.transform(X[X_features_cat])

    #reorder the columns
    X = X[['GAME_ID', 'GAME_DATE', 'TEAM_ID_h', 'TEAM_NAME_h',
       'TEAM_ABBREVIATION_h', 'HOME_TEAM_h', 'OFF_RATING_h', 'DEF_RATING_h',
       'NET_RATING_h', 'AST_PCT_h', 'AST_TOV_h', 'OREB_PCT_h', 'DREB_PCT_h',
       'REB_PCT_h', 'TM_TOV_PCT_h', 'EFG_PCT_h', 'TS_PCT_h', 'PACE_h',
       'POSS_h', 'PTS_h', 'TEAM_ID_a', 'TEAM_NAME_a', 'TEAM_ABBREVIATION_a',
       'HOME_TEAM_a', 'OFF_RATING_a', 'DEF_RATING_a', 'NET_RATING_a',
       'AST_PCT_a', 'AST_TOV_a', 'OREB_PCT_a', 'DREB_PCT_a', 'REB_PCT_a',
       'TM_TOV_PCT_a', 'EFG_PCT_a', 'TS_PCT_a', 'PACE_a', 'POSS_a', 'PTS_a',
       'ATL_h', 'BKN_h', 'BOS_h', 'CHA_h', 'CHI_h', 'CLE_h',
       'DAL_h', 'DEN_h', 'DET_h', 'GSW_h', 'HOU_h', 'IND_h', 'LAC_h', 'LAL_h',
       'MEM_h', 'MIA_h', 'MIL_h', 'MIN_h', 'NOP_h', 'NYK_h', 'OKC_h', 'ORL_h',
       'PHI_h', 'PHX_h', 'POR_h', 'SAC_h', 'SAS_h', 'TOR_h', 'UTA_h', 'WAS_h',
       'ATL_a', 'BKN_a', 'BOS_a', 'CHA_a', 'CHI_a', 'CLE_a', 'DAL_a', 'DEN_a',
       'DET_a', 'GSW_a', 'HOU_a', 'IND_a', 'LAC_a', 'LAL_a', 'MEM_a', 'MIA_a',
       'MIL_a', 'MIN_a', 'NOP_a', 'NYK_a', 'OKC_a', 'ORL_a', 'PHI_a', 'PHX_a',
       'POR_a', 'SAC_a', 'SAS_a', 'TOR_a', 'UTA_a', 'WAS_a', 'PLUS_MINUS']]

    #define the features
    X_features = ['HOME_TEAM_h', 'OFF_RATING_h', 'DEF_RATING_h',
       'NET_RATING_h', 'AST_PCT_h', 'AST_TOV_h', 'OREB_PCT_h', 'DREB_PCT_h',
       'REB_PCT_h', 'TM_TOV_PCT_h', 'EFG_PCT_h', 'TS_PCT_h', 'PACE_h', 'POSS_h',
       'HOME_TEAM_a', 'OFF_RATING_a', 'DEF_RATING_a', 'NET_RATING_a',
       'AST_PCT_a', 'AST_TOV_a', 'OREB_PCT_a', 'DREB_PCT_a', 'REB_PCT_a',
       'TM_TOV_PCT_a', 'EFG_PCT_a', 'TS_PCT_a', 'PACE_a', 'POSS_a',
       'ATL_h', 'BKN_h', 'BOS_h', 'CHA_h', 'CHI_h', 'CLE_h', 'DAL_h', 'DEN_h',
       'DET_h', 'GSW_h', 'HOU_h', 'IND_h', 'LAC_h', 'LAL_h', 'MEM_h', 'MIA_h',
       'MIL_h', 'MIN_h', 'NOP_h', 'NYK_h', 'OKC_h', 'ORL_h', 'PHI_h', 'PHX_h',
       'POR_h', 'SAC_h', 'SAS_h', 'TOR_h', 'UTA_h', 'WAS_h', 'ATL_a', 'BKN_a',
       'BOS_a', 'CHA_a', 'CHI_a', 'CLE_a', 'DAL_a', 'DEN_a', 'DET_a', 'GSW_a',
       'HOU_a', 'IND_a', 'LAC_a', 'LAL_a', 'MEM_a', 'MIA_a', 'MIL_a', 'MIN_a',
       'NOP_a', 'NYK_a', 'OKC_a', 'ORL_a', 'PHI_a', 'PHX_a', 'POR_a', 'SAC_a',
       'SAS_a', 'TOR_a', 'UTA_a', 'WAS_a']

    #define and return X and y
    X_preproc = X[X_features]
    y = X['PLUS_MINUS']

    return X_preproc, y
