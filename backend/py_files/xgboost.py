import datetime
import pandas as pd
import seaborn as sns
import numpy as np
import datetime as dt

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

import xgboost as xgb
from hyperopt import Trials, STATUS_OK, tpe, hp, fmin

def load_data(path = 'backend/data/pkl/scaled_boxscore_advanced_rolling_players.pkl'):
    df = pd.read_pickle(path)
    return df

def train_test_split(df, train_test_split_date = '2022-04-27 00:00:00'):
    X_train = df.loc[df.GAME_DATE < train_test_split_date, ~df.columns.isin(['PLUS_MINUS'])]
    X_test = df.loc[df.GAME_DATE > train_test_split_date, ~df.columns.isin(['PLUS_MINUS'])]
    y_train = df.loc[df.GAME_DATE < train_test_split_date, ['PLUS_MINUS']]
    y_test = df.loc[df.GAME_DATE > train_test_split_date, ['PLUS_MINUS']]
    return X_train, X_test, y_train, y_test

def remove_categorical(X_train, X_test):
    X_train_catless = X_train.loc[:, ~X_train.columns.isin(['PLAYER_ID', 'TEAM_ID', 'TEAM_ABBREVIATION', 'GAME_DATE', 'PLAYER_ID', 'PLAYER_NAME'])]
    X_test_catless = X_test.loc[:, ~X_test.columns.isin(['PLAYER_ID', 'TEAM_ID', 'TEAM_ABBREVIATION', 'GAME_DATE', 'PLAYER_ID', 'PLAYER_NAME'])]
    return X_train_catless, X_test_catless

space = {
    'max_depth' : hp.choice('max_depth', range(5, 100, 1)),
    'learning_rate' : hp.quniform('learning_rate', 0.01, 0.5, 0.01),
    'n_estimators' : hp.choice('n_estimators', range(20, 600, 5)),
    'gamma' : hp.quniform('gamma', 0, 0.50, 0.01),
    'min_child_weight' : hp.quniform('min_child_weight', 1, 10, 1),
    'subsample' : hp.quniform('subsample', 0.1, 1, 0.01),
    'colsample_bytree' : hp.quniform('colsample_bytree', 0.1, 1.0, 0.01)
}


def hyperopt_objective(space, X_train_catless, y_train):
    '''
    Defines the ML model to be optimised. In this case, we use a hyperopt regressor.
    space: the hyperparameter search space. A hyperopt space object, ie a dict where each entry consists of
    'feature' : hp.trial_value_selection_rule('feature', [trial values])
    '''
    xgb.warnings.filterwarnings(action='ignore', category=DeprecationWarning)
    hyperopt_regressor = xgb.XGBRegressor(n_estimators = space['n_estimators'],
                            max_depth = int(space['max_depth']),
                            learning_rate = space['learning_rate'],
                            gamma = space['gamma'],
                            min_child_weight = space['min_child_weight'],
                            subsample = space['subsample'],
                            colsample_bytree = space['colsample_bytree'],
                            objective = 'reg:pseudohubererror'
                            )

    hyperopt_regressor.fit(X_train_catless, y_train)

    # Applying k-Fold Cross Validation
    mse = cross_val_score(estimator = hyperopt_regressor, X = X_train_catless, y = y_train, cv = 10, n_jobs=-1)
    CrossValMSE = mse.mean()

    print("CrossValMSE: ", CrossValMSE)
    return{'loss': CrossValMSE, 'status': STATUS_OK }


def hyperparam_optimise(hyperopt_objective, space, algo=tpe.suggest, max_evals=100):
    '''
    Optimises the objective function.
    hyperopt_objective: the hyperopt_objective function
    space: the hyperparameter search space. A hyperopt space object, ie a dict where each entry consists of
    'feature' : hp.trial_value_selection_rule('feature', [trial values])
    algo: the hyperopt search algorithm. tpe.suggest recommended
    max_evals: the maximum number of optimisation trials
    '''
    trials = Trials()
    best = fmin(fn=hyperopt_objective,
                space=space,
                algo=tpe.suggest,
                max_evals=100,
                trials=trials)

    print("Best: ", best)
    return best


def train_model(X_train_catless, y_train, best):
    regressor = xgb.XGBRegressor(n_estimators = best['n_estimators'],
                                    max_depth = best['max_depth'],
                                    learning_rate = best['learning_rate'],
                                    gamma = best['gamma'],
                                    min_child_weight = best['min_child_weight'],
                                    subsample = best['subsample'],
                                    colsample_bytree = best['colsample_bytree'],
                                    objective='reg:squarederror'
                                    )

    regressor.fit(X_train_catless, y_train)
    return regressor

def load_pretrained_model(regressor = xgb.XGBRegressor(), path = 'nba_betting_analysis/backend/models/hyperopted_gbdt_players_33estimators')
    return regressor.load_model(path)

def score_model(regressor, X_test_catless, y_test):
    y_pred = regressor.predict(X_test_catless)
    r2 = regressor.score(X_test_catless, y_test)
    rmse = np.sqrt(mean_squared_error(y_test.values, y_pred))
    print(r2, rmse)


def predict_player_plus_minus(regressor, X_test_catless):
    return regressor.predict(X_test_catless)


def add_predicted_scores_to_df_for_team_aggregation(df, X_train, X_test, X_train_catless, X_test_catless, regressor):
    '''
    Returns the advanced player boxscores df, then adds the predicted train and test values
    along with a Boolean column indicating whether the row is a train or test row.
    This is done for the next step, which is aggregating player predictions into team predictions.
    '''
    df_train = df.loc[df.index.isin(X_train.index), :].copy()
    df_test = df.loc[df.index.isin(X_test.index), :].copy()
    df_train['y_pred'] = regressor.predict(X_train_catless)
    df_test['y_pred'] = regressor.predict(X_test_catless)
    df = pd.concat((df_train, df_test), axis=0)
    df['IS_TEST_ROW'] = 0
    df.loc[df.index.isin(X_test.index), ['IS_TEST_ROW']] = 1
    return df

def aggregate_players_into_teams_for_metalearning(df):
    '''
    Aggregates the predicted player plus/minus score contributions into whole team score contributions.
    These can then feed a metalearner that takes this data, and the modelled team data
    '''
    grouping_df = df.loc[:, ['PLAYER_ID', 'TEAM_ID', 'BETTING_TABLE_TEAM_NAME', 'GAME_DATE', 'PLAYER_NAME', 'PLUS_MINUS', 'y_pred']].copy()
    grouping_df.groupby(by=['BETTING_TABLE_TEAM_NAME', 'GAME_DATE'], as_index=False)['PLUS_MINUS', 'y_pred'].sum()
    return grouping_df
