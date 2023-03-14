import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def create_pysbr_team_names(nba_path, pysbr_path):
    '''takes pickle files and preprocess them for easy comparison of sports betting team naming format'''
    df_sbr = pd.read_pickle(pysbr_path)
    df_teamnames = pd.read_pickle(nba_path)
    df_teamnames = df_teamnames[['TEAM_NAME', 'TEAM_ABBREVIATION']]

    #----------------maps participant name into the nba format----------------------------------------
    df_sbr = df_sbr[['participants.1.participant id', 'participants.1.source.nickname',
       'participants.1.source.short name', 'participants.2.participant id',
       'participants.2.source.nickname', 'participants.2.source.short name']]
    df_sbr['p1_full'] = df_sbr['participants.1.source.short name'] + ' ' + df_sbr['participants.1.source.nickname']
    df_sbr['p2_full'] = df_sbr['participants.2.source.short name'] + ' ' + df_sbr['participants.2.source.nickname']
    df_sbr['p2_full'] = df_sbr['p2_full'].replace({'L.A. Clippers L.A. Clippers': 'LA Clippers', 'L.A. Lakers L.A. Lakers': 'Los Angeles Lakers'})
    df_sbr['p1_full'] = df_sbr['p1_full'].replace({'L.A. Clippers L.A. Clippers': 'LA Clippers', 'L.A. Lakers L.A. Lakers': 'Los Angeles Lakers'})
    df_sbr['participants.2.source.short name'] = df_sbr['participants.2.source.short name'].replace({'L.A. Clippers': 'Los Angeles ', 'L.A. Lakers': 'Los Angeles'})
    df_sbr['participants.1.source.short name'] = df_sbr['participants.1.source.short name'].replace({'L.A. Clippers': 'Los Angeles ', 'L.A. Lakers': 'Los Angeles'})
    df_sbr = df_sbr[df_sbr['p1_full'].isin(pd.unique(df_teamnames['TEAM_NAME']))]
    df_sbr = df_sbr[df_sbr['p2_full'].isin(pd.unique(df_teamnames['TEAM_NAME']))]
    full_names_participant = df_sbr.drop_duplicates('participants.1.participant id')
    full_names_participant['participant id'] = full_names_participant['participants.1.participant id'].copy()
    full_names_participant['TEAM_NAME'] = full_names_participant['p1_full'].copy()
    full_temp = full_names_participant[['participant id','TEAM_NAME', 'participants.1.source.short name']]
    full_name_abb = pd.merge(full_temp, df_teamnames, on='TEAM_NAME').drop_duplicates('participant id')
    full_name_abb['HOME_LOCATION'] = full_name_abb['participants.1.source.short name']
    full_name_abb = full_name_abb[['participant id', 'TEAM_NAME', 'TEAM_ABBREVIATION', 'HOME_LOCATION']]
    #---------------------------------------------------------------------------------------------------
    home_name_abb = full_name_abb[['TEAM_NAME', 'TEAM_ABBREVIATION']]
    home_name_abb.rename(columns={'TEAM_NAME':'home_team'}, inplace=True)

    away_name_abb = full_name_abb[['TEAM_NAME', 'TEAM_ABBREVIATION']]
    away_name_abb.rename(columns={'TEAM_NAME':'away_team'}, inplace=True)

    return (home_name_abb, away_name_abb, full_name_abb)

def create_mergeable_pysbr(nba_path, pysbr_path):
    '''extracts betting odds for mergeable pysbr'''
    home_name_abb, away_name_abb, full_name_abb = create_pysbr_team_names(nba_path, pysbr_path)
    df_sbr = pd.read_pickle(pysbr_path).copy()
    df_nba = pd.read_pickle(nba_path).copy()
    df_nba['GAME_DATE'] = pd.to_datetime(df_nba['GAME_DATE'], format='%Y-%M-%D')

    merged_sbr_df = pd.merge(df_sbr, full_name_abb, on='participant id')
    merged_sbr_df['GAME_DATE'] = merged_sbr_df['datetime_x']
    merged_sbr_df.drop(['datetime_x'], axis=1, inplace=True)
    merged_sbr_df['HOME_TEAM'] = np.where(merged_sbr_df['HOME_LOCATION']==merged_sbr_df['location'], 1, 0)
    selected_sbr_df = merged_sbr_df[['event id', 'TEAM_NAME','TEAM_ABBREVIATION', 'HOME_LOCATION', 'GAME_DATE', 'HOME_TEAM', 'sportsbook_name', 'participant id',
                               'description', 'location', 'decimal odds', 'american odds', 'spread / total']]

    selected_sbr_df[['away_team', 'home_team']] = selected_sbr_df['description'].str.split('@', expand=True)
    selected_sbr_df = pd.merge(selected_sbr_df, home_name_abb, on='home_team')
    selected_sbr_df = pd.merge(selected_sbr_df, away_name_abb, on='away_team')
    selected_sbr_df.rename(columns={'TEAM_ABBREVIATION_y': 'TEAM_ABBREVIATION_h', 'TEAM_ABBREVIATION':'TEAM_ABBREVIATION_a', 'TEAM_ABBREVIATION_x':'TEAM_ABBREVIATION'}, inplace=True)

    selected_sbr_df['MATCHUP'] = np.where(selected_sbr_df['HOME_TEAM']==1, selected_sbr_df['TEAM_ABBREVIATION_h'] + ' vs. ' + selected_sbr_df['TEAM_ABBREVIATION_a'],
                                          selected_sbr_df['TEAM_ABBREVIATION_a'] + ' @ ' + selected_sbr_df['TEAM_ABBREVIATION_h'])

    selected_sbr_df['GAME_DATE'] =  pd.to_datetime(selected_sbr_df['GAME_DATE']).dt.date
    selected_sbr_df['GAME_DATE'] =  pd.to_datetime(selected_sbr_df['GAME_DATE'])
    output_df = pd.merge(df_nba, selected_sbr_df[['GAME_DATE', 'sportsbook_name', 'MATCHUP', 'TEAM_ABBREVIATION_a', 'TEAM_ABBREVIATION_h', 'spread / total', 'decimal odds', 'american odds']], on=
                         ['GAME_DATE', 'MATCHUP'])

    return output_df[['GAME_ID', 'TEAM_ABBREVIATION', 'GAME_DATE', 'sportsbook_name', 'spread / total', 'decimal odds', 'american odds']].sort_values('GAME_DATE')

def prepping_model_data(preprocessed_path, columns_remove=None):

    '''OHE method only. Returns (X,y_ train and test set
    before splitting works for preprocessed data such as ADV_OHE_TEAM_ALL,
    removes uneccessary columns for analysis'''

    df_train = pd.read_pickle(preprocessed_path)
    #set target_variable
    df_target = df_train['PLUS_MINUS']
    #removes unnecessary data columns
    columns_drop = ['PLUS_MINUS']
    df_X = df_train.drop(columns_drop, axis=1)
    #drops additional columns after feature importance
    if columns_remove is not None:
        df_X.drop(columns_remove, axis=1, inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(df_X, df_target, random_state=0, test_size=0.3)
    return (X_train, X_test, y_train, y_test)

def scale_y(y_train, y_test):
    '''minmax y, returns scaler for inverse transforms and the transformed y train and test set'''
    scaler = MinMaxScaler()
    y_train = scaler.fit_transform(np.array(y_train).reshape(-1, 1))
    y_test = scaler.transform(np.array(y_test).reshape(-1, 1))
    return (y_train, y_test, scaler)

def fit_XGboost_huber(X_train, y_train, sub_sample= 0.39, regression_lambda= 0.028299,regression_alpha=0.033299,
                number_estimators= 650, maximum_depth=32, lr = 0.021399, huber_s=4.56):
    '''fit our best XGboost from randomised search cv, you can replace the parameters. Using sklearn based interface'''

    columns_drop = ['GAME_ID', 'GAME_DATE', 'TEAM_ID_h', 'TEAM_NAME_h', 'TEAM_ABBREVIATION_h', 'TEAM_ID_a','TEAM_NAME_a','TEAM_ABBREVIATION_a', 'HOME_TEAM_h', 'HOME_TEAM_a']
    X_train.drop(columns_drop, axis=1, inplace=True)
    xg_model = XGBRegressor(subsample= sub_sample, reg_lambda=regression_lambda,reg_alpha=regression_alpha, objective='reg:pseudohubererror', n_estimators=number_estimators, max_depth=maximum_depth,
                        learning_rate=lr, huber_slope=huber_s, n_jobs=-1)
    xg_model.fit(X_train, y_train)
    return xg_model

def cross_validate_model(model, X_train, y_train, folds=5):
    '''cross validate model works for only sklearn models'''
    score = cross_val_score(model, X_train, y_train, cv=folds)
    return np.mean(score)

def merge_odds_test_set(model, sbr_processed, X_test, y_test, scaler=None):
    '''making comparison of betting odds and predicted data, test set reccomended otherwise it wont be a fair comparison.
    Do note if Y is scaled scaler argument must be present, the scaler must be already fitted
    sbr processed from create_mergeable_pysbr'''

    if scaler is not None:
        y_test = scaler.inverse_transform(y_test)
    df_test = X_test.copy()
    df_test['PLUS_MINUS'] = y_test
    columns_drop = ['GAME_ID', 'GAME_DATE', 'TEAM_ID_h', 'TEAM_NAME_h', 'TEAM_ABBREVIATION_h', 'TEAM_ID_a','TEAM_NAME_a','TEAM_ABBREVIATION_a', 'HOME_TEAM_h', 'HOME_TEAM_a']
    temp_predict = X_test.drop(columns_drop, axis=1)

    y_predict = model.predict(temp_predict)
    if scaler is not None:
        df_test['y_pred'] = scaler.inverse_transform(y_predict.reshape(-1, 1))
    else:
        df_test['y_pred'] = y_predict
    #df_merged = pd.merge(df_test, sbr_processed[['GAME_ID', 'TEAM_ABBREVIATION', 'sportsbook_name', 'spread / total', 'decimal odds', 'american odds']], on='GAME_ID', how='inner')
    #
    return df_test

if __name__ == "__main__":
    #getting sbr_processed
    nba_path = '../data/pkl/raw_games_5yrs.pkl'
    pysbr_path = '../data/pkl/pysbr_raw.pkl'
    home_name_abb, away_name_abb, full_name_abb = create_pysbr_team_names(nba_path, pysbr_path)
    output_df = create_mergeable_pysbr(nba_path, pysbr_path)

    #example with model with XGboost
    col_remove = ['OFF_RATING_h','DEF_RATING_h','NET_RATING_h',
     'AST_PCT_h','AST_TOV_h','OREB_PCT_h','DREB_PCT_h','REB_PCT_h',
     'TM_TOV_PCT_h','EFG_PCT_h','TS_PCT_h','PACE_h','POSS_h','OFF_RATING_a',
     'DEF_RATING_a','NET_RATING_a','AST_PCT_a','AST_TOV_a','OREB_PCT_a',
     'DREB_PCT_a','REB_PCT_a','TM_TOV_PCT_a','EFG_PCT_a','TS_PCT_a',
     'PACE_a','POSS_a', 'AST_PCT_Roll_std_h','AST_PCT_Roll_median_a','REB_PCT_Roll_std_h',
     'TM_TOV_PCT_Roll_std_h','AST_PCT_Roll_std_a','OREB_PCT_Roll_median_h','OREB_PCT_Roll_std_a',
     'EFG_PCT_Roll_std_h','POSS_Roll_std_a','AST_TOV_Roll_median_h','OREB_PCT_Roll_median_a',
     'AST_PCT_Roll_mean_a','OREB_PCT_Roll_std_h','AST_TOV_Roll_std_a','PACE_Roll_std_h',
     'DREB_PCT_Roll_std_h','REB_PCT_Roll_median_h','PACE_Roll_median_h','TS_PCT_Roll_mean_h','REB_PCT_Roll_mean_h',
     'POSS_Roll_mean_h','AST_PCT_Roll_median_h','EFG_PCT_Roll_median_a','DREB_PCT_Roll_median_h',
     'EFG_PCT_Roll_mean_h','PACE_Roll_mean_h','TM_TOV_PCT_Roll_mean_h','DREB_PCT_Roll_mean_h',
     'OREB_PCT_Roll_mean_h','AST_TOV_Roll_mean_h','AST_PCT_Roll_mean_h','DEF_RATING_Roll_mean_h','OFF_RATING_Roll_mean_h']

    prep_path = "../data/pkl/ADV_OHE_TEAM_ALL"

    X_train, X_test, y_train, y_test = prepping_model_data(prep_path, columns_remove=col_remove)
    print(X_train.shape)
    print(X_test.shape)
    y_train, y_test, scaler = scale_y(y_train, y_test)
    model = fit_XGboost_huber(X_train, y_train)
    #print(cross_validate_model(model, X_train, y_train, folds=5))
    var = merge_odds_test_set(model, output_df, X_test, y_test, scaler)
    var.to_pickle('y_pred_adv_test_alec.pkl')
