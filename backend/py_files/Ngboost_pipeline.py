import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
import pickle
from pathlib import Path
from ngboost import NGBRegressor
from sklearn.tree import DecisionTreeRegressor
from scipy.integrate import quad
import seaborn as sns
import scipy

def get_preproc_ngboost(path, split_date='2022-04-27 00:00:00'):
    '''removed columns for ngboost, returns X and y train and test set.
    similar to sklearn train test split. Format needs to be ADV_OHE_TEAM_ALL'''
    df = pd.read_pickle(path)

    #------------------------------------------------------------------------
    columns_drop = ['GAME_ID', 'TEAM_ID_h', 'TEAM_ABBREVIATION_h', 'TEAM_ID_a','TEAM_NAME_a','TEAM_ABBREVIATION_a', 'HOME_TEAM_h', 'HOME_TEAM_a', 'OFF_RATING_h','DEF_RATING_h','NET_RATING_h','AST_PCT_h','AST_TOV_h','OREB_PCT_h','DREB_PCT_h','REB_PCT_h','TM_TOV_PCT_h','EFG_PCT_h','TS_PCT_h','PACE_h','POSS_h','OFF_RATING_a','DEF_RATING_a','NET_RATING_a','AST_PCT_a','AST_TOV_a','OREB_PCT_a','DREB_PCT_a','REB_PCT_a','TM_TOV_PCT_a','EFG_PCT_a','TS_PCT_a','PACE_a','POSS_a']
    df = df.drop(columns_drop, axis=1)
    #target = df['PLUS_MINUS']

    train_test_split_date = split_date
    X_train = df.loc[df.GAME_DATE < train_test_split_date, ~df.columns.isin(['PLUS_MINUS'])]
    X_test = df.loc[df.GAME_DATE > train_test_split_date, ~df.columns.isin(['PLUS_MINUS'])]
    y_train = df.loc[df.GAME_DATE < train_test_split_date, ['PLUS_MINUS']]['PLUS_MINUS']
    y_test = df.loc[df.GAME_DATE > train_test_split_date, ['PLUS_MINUS']]['PLUS_MINUS']
    X_train.drop(['TEAM_NAME_h','GAME_DATE'], axis=1, inplace=True)
    X_test.drop(['TEAM_NAME_h','GAME_DATE'], axis=1, inplace=True)

    #----------removed features based on feature importance------------------
    importances_removed = ['POSS_Roll_median_h', 'OFF_RATING_Roll_std_a', 'PACE_Roll_mean_a', 'REB_PCT_Roll_std_h', 'AST_PCT_Roll_std_a', 'AST_TOV_Roll_mean_h', 'EFG_PCT_Roll_mean_a', 'OFF_RATING_Roll_median_a', 'TS_PCT_Roll_std_a', 'REB_PCT_Roll_mean_a', 'OFF_RATING_Roll_std_h', 'OFF_RATING_Roll_mean_a', 'OREB_PCT_Roll_mean_a', 'REB_PCT_Roll_median_a', 'TM_TOV_PCT_Roll_std_h', 'OREB_PCT_Roll_median_h', 'TS_PCT_Roll_std_h', 'EFG_PCT_Roll_std_h', 'DREB_PCT_Roll_std_h', 'AST_PCT_Roll_median_h', 'DEF_RATING_Roll_std_h', 'AST_PCT_Roll_mean_a', 'TM_TOV_PCT_Roll_mean_h', 'OREB_PCT_Roll_median_a', 'AST_TOV_Roll_median_h', 'POSS_Roll_median_a', 'DEF_RATING_Roll_median_h', 'DREB_PCT_Roll_median_h', 'EFG_PCT_Roll_std_a', 'PACE_Roll_median_a', 'DREB_PCT_Roll_mean_h', 'POSS_Roll_mean_h', 'TM_TOV_PCT_Roll_median_a', 'TM_TOV_PCT_Roll_median_h', 'PACE_Roll_median_h', 'DREB_PCT_Roll_mean_a', 'POSS_Roll_mean_a', 'TS_PCT_Roll_median_h', 'TM_TOV_PCT_Roll_std_a', 'POSS_Roll_std_a', 'DEF_RATING_Roll_median_a', 'EFG_PCT_Roll_median_h', 'OFF_RATING_Roll_median_h', 'EFG_PCT_Roll_median_a', 'TS_PCT_Roll_mean_a', 'OREB_PCT_Roll_std_h', 'NET_RATING_Roll_std_a', 'POSS_Roll_std_h', 'AST_PCT_Roll_mean_h', 'PACE_Roll_mean_h', 'OREB_PCT_Roll_std_a', 'DREB_PCT_Roll_std_a', 'REB_PCT_Roll_std_a', 'PACE_Roll_std_a', 'AST_PCT_Roll_median_a', 'REB_PCT_Roll_mean_h', 'REB_PCT_Roll_median_h']

    X_train_fs = X_train.drop(importances_removed, axis=1)
    X_test_fs = X_test.drop(importances_removed, axis=1)

    return (X_train_fs, X_test_fs, y_train, y_test)

def get_new_preproc_ngboost(path):
    df = pd.read_pickle(path)
    columns_drop = ['GAME_ID', 'TEAM_ID_h', 'TEAM_ABBREVIATION_h', 'TEAM_ID_a','TEAM_NAME_a','TEAM_ABBREVIATION_a', 'HOME_TEAM_h', 'HOME_TEAM_a', 'OFF_RATING_h','DEF_RATING_h','NET_RATING_h','AST_PCT_h','AST_TOV_h','OREB_PCT_h','DREB_PCT_h','REB_PCT_h','TM_TOV_PCT_h','EFG_PCT_h','TS_PCT_h','PACE_h','POSS_h','OFF_RATING_a','DEF_RATING_a','NET_RATING_a','AST_PCT_a','AST_TOV_a','OREB_PCT_a','DREB_PCT_a','REB_PCT_a','TM_TOV_PCT_a','EFG_PCT_a','TS_PCT_a','PACE_a','POSS_a']
    X = df.drop(columns_drop, axis=1)
    y = df['PLUS_MINUS']

    X.drop(['TEAM_NAME_h','GAME_DATE', 'PLUS_MINUS'], axis=1, inplace=True)
    importances_removed = ['POSS_Roll_median_h', 'OFF_RATING_Roll_std_a', 'PACE_Roll_mean_a', 'REB_PCT_Roll_std_h', 'AST_PCT_Roll_std_a', 'AST_TOV_Roll_mean_h', 'EFG_PCT_Roll_mean_a', 'OFF_RATING_Roll_median_a', 'TS_PCT_Roll_std_a', 'REB_PCT_Roll_mean_a', 'OFF_RATING_Roll_std_h', 'OFF_RATING_Roll_mean_a', 'OREB_PCT_Roll_mean_a', 'REB_PCT_Roll_median_a', 'TM_TOV_PCT_Roll_std_h', 'OREB_PCT_Roll_median_h', 'TS_PCT_Roll_std_h', 'EFG_PCT_Roll_std_h', 'DREB_PCT_Roll_std_h', 'AST_PCT_Roll_median_h', 'DEF_RATING_Roll_std_h', 'AST_PCT_Roll_mean_a', 'TM_TOV_PCT_Roll_mean_h', 'OREB_PCT_Roll_median_a', 'AST_TOV_Roll_median_h', 'POSS_Roll_median_a', 'DEF_RATING_Roll_median_h', 'DREB_PCT_Roll_median_h', 'EFG_PCT_Roll_std_a', 'PACE_Roll_median_a', 'DREB_PCT_Roll_mean_h', 'POSS_Roll_mean_h', 'TM_TOV_PCT_Roll_median_a', 'TM_TOV_PCT_Roll_median_h', 'PACE_Roll_median_h', 'DREB_PCT_Roll_mean_a', 'POSS_Roll_mean_a', 'TS_PCT_Roll_median_h', 'TM_TOV_PCT_Roll_std_a', 'POSS_Roll_std_a', 'DEF_RATING_Roll_median_a', 'EFG_PCT_Roll_median_h', 'OFF_RATING_Roll_median_h', 'EFG_PCT_Roll_median_a', 'TS_PCT_Roll_mean_a', 'OREB_PCT_Roll_std_h', 'NET_RATING_Roll_std_a', 'POSS_Roll_std_h', 'AST_PCT_Roll_mean_h', 'PACE_Roll_mean_h', 'OREB_PCT_Roll_std_a', 'DREB_PCT_Roll_std_a', 'REB_PCT_Roll_std_a', 'PACE_Roll_std_a', 'AST_PCT_Roll_median_a', 'REB_PCT_Roll_mean_h', 'REB_PCT_Roll_median_h']
    X_fs = X.drop(importances_removed, axis=1)

    return (X_fs, y)

def update_ngboost(X_train, y_train):
    '''return the newly update ngboost model'''

    base=DecisionTreeRegressor(criterion='friedman_mse', max_depth=5)
    ngb = NGBRegressor(col_sample=0.4, minibatch_frac=0.2,
                 n_estimators=485, Base=base, learning_rate=0.01)
    y_train = y_train.ravel()
    weights_loc = np.argwhere((y_train < -15) | (y_train > 15))
    sample_weight = np.array([1] * y_train.shape[0])
    sample_weight[weights_loc.ravel()] += 1
    for i in range(5):
        ngb.fit(X_train.reset_index(drop=True), y_train, sample_weight=sample_weight)
    return ngb

def load_ngboost_team_model(file_path):
    '''use path.home to reach data/pkl folder, this returns the
    model from the pkl file'''
    with file_path.open("rb") as f:
        ngb_unpickled = pickle.load(f)
    return ngb_unpickled

def visualise_distribution(model, X_test, game_index:int):
    '''accepts the ng boost model, and the distribution of predicted values. Specify the game_index for one distribution. X_test can be you split model or the new data you want to predict'''
    Y_dist = model.pred_dist(X_test)
    value = np.random.normal(loc=Y_dist.params['loc'][game_index],
                             scale=Y_dist.params['scale'][game_index],size=1000)
    sns.histplot(value, kde=True)

def get_mean_std_normal_distribution(model, X_test):
    '''returns the normal distribution mean and standard deviation of games by their index
    from the Y_dist params loc and scale respectively given the NGboost model.
    '''
    Y_dist = model.pred_dist(X_test)

    mean = Y_dist.params['loc']
    std = Y_dist.params['scale']

    return (mean, std)

def normal_distribution_function(x, mean, std):
    '''returns the point probability of the normal distribution of a given PLUS_MINUS'''
    value = scipy.stats.norm.pdf(x,loc=mean,scale=std)
    return value

def single_range_distribution_prediction(mean, std, x1=0,x2=100):
    '''returns the probability over the range of values.
    defaults to the right side of the distribution'''
    return quad(normal_distribution_function, x1, x2, args=(mean, std))

def get_left_betting_metric(x_min, x_max, mean, std, percentile=0.54):
    '''gets_the_cumulative_sum of PLUS_MINUS probability until the Nth percentile distribution
    (cumulative sum from the left side)'''
    x = np.linspace(x_min, x_max, 1000)
    cum_sum = scipy.stats.norm.cdf(x, mean, std)
    bet_left = np.argwhere(np.isclose(cum_sum, percentile, rtol=1e-02, atol=1e-04))[0][0]
    return np.round(x[bet_left])

def get_right_betting_metric(x_min, x_max, mean, std, percentile=0.54):
    '''gets_the_cumulative_sum of PLUS_MINUS probability until the Nth percentile
    (cumulative sum from the right side)'''
    x = np.linspace(x_min, x_max, 1000)
    cum_sum = scipy.stats.norm.cdf(x, mean, std)
    bet_right = np.argwhere(np.isclose(1-cum_sum, percentile, rtol=1e-02, atol=1e-04))[0][0]
    return np.round(x[bet_right])

def get_all_range_distribution_prediction(model, X_test, y_test):
    '''returns the probability of the PLUS_MINUS on multiple games.
    defaults to the right side of the distribution. Over large data and
    returns a list. X_test can be the new data, y_test can also be the sbr's
    actual metrics'''
    dist_prob = []
    means, stdevs = get_mean_std_normal_distribution(model, X_test)

    #loops over different games
    for i in range(y_test.shape[0]):
        mean = means[i]
        std = stdevs[i]
        if y_test[i] >= 0:
            dist_prob.append(quad(normal_distribution_function, y_test[i], max(y_test), args=(mean, std))[0])
        elif y_test[i] < 0:
            dist_prob.append(quad(normal_distribution_function, min(y_test), y_test[i], args=(mean, std))[0])
    return pd.DataFrame({'y_true':y_test, 'probability':dist_prob})

if __name__ == "__main__":
    path = '../data/pkl/ADV_OHE_TEAM_ALL'
    #get data
    X_train, X_test, y_train, y_test = get_preproc_ngboost(path, split_date='2022-04-27 00:00:00')

    #load model as unpickle ngboost
    file_path = Path.home()/'code'/'alecmatt5'/'nba_betting_analysis'/'backend'/'data'/'pkl'/'ngdemo.pkl'
    unpickle_ngb = load_ngboost_team_model(file_path)

    #visualise distributions of PLUS_MINUS of each game, modify the interger
    #at the third param of visualise_distribution
    visualise_distribution(unpickle_ngb, X_test_fs, 0)

    #get arrays of mean and std of different games specified at X_test
    mean, std = get_mean_std_normal_distribution(unpickle_ngb, X_test)

    #get the probability at a single point
    normal_distribution_function(5, mean[0], std[0])

    #get the probability of the range of values
    single_range_distribution_prediction(mean[0], std[0], x1=0,x2=max(y_test))

    #get the probability of the range of values based on the cumulative percentile
    #cum_sum from the left side
    get_left_betting_metric(min(y_test), max(y_test), mean[0], std[0], percentile=0.54)
    #get the probability of the range of values based on the cumulative percentile
    #cum_sum from the right side
    get_right_betting_metric(min(y_test), max(y_test), mean[0], std[0], percentile=0.54)

    #get all range of probability distribution of a given
    dist_prob = get_all_range_distribution_prediction(unpickle_ngb, X_test, y_test)
