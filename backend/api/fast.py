from py_files.ngboost_team import get_y_pred_percentile_api
from py_files.ngboost_team import load_ngboost_team_model
from py_files.today_games_preprocessor import preprocess_advanced
#from py_files.bet_scoring import *
from pathlib import Path
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import glob

#today_date = datetime.today().strftime('%Y-%m-%d')
today_date = glob.glob('data/pkl/demo*.pkl')[-1]
print(today_date)
app = FastAPI()
file_path_model = 'data/pkl/ngdemo.pkl'
app.state.model = load_ngboost_team_model(file_path_model)
app.state.df_app = pd.read_pickle(today_date).copy()
#app.state.df_bets = get_today_betting_data('data/pkl/sbr_today_betting_data.pkl')
# app.state.df_app, X_features = preprocess_advanced('boxscores_advanced_team_all.pkl',
#                                         roll_methods=['mean', 'median', 'std'],
#                                         ohe=True,
#                                         scaled=False)
# Optional, good practice for dev purposes. Allow all middlewares
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=False,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
# @app.get("/predict")
# def predict(percentile_target=0.54):
#     percentile_target = float(percentile_target)
#     bets = app.state.df_bets.copy()
#     model = app.state.model
#     df_app=app.state.df_app.copy()
#     y_pred = get_y_pred_percentile_api(df_app, model,
#                                             percentile=percentile_target)
#     compare = merge_today_predictions_with_bet_data(y_pred=y_pred, bets=bets)
#     predicted_bet = predict_today_bets(compare=compare)
#     y_json = predicted_bet.to_dict()
#     return y_json

@app.get("/predict")
def predict(percentile_target=0.54):
    percentile_target = float(percentile_target)
    model = app.state.model
    df_app=app.state.df_app.copy()
    y_pred = get_y_pred_percentile_api(df_app, model,
                                            percentile=percentile_target)
    return y_pred


@app.get("/")
def root():
    X_json = app.state.df_app.to_dict()
    return X_json

if __name__=='__main__':
    print(app.state.df_app)
    y_json = predict(percentile_target=0.6)
    print(y_json)
    y_json.to_pickle('data/pkl/DEMO_PRED.pkl')
    #print(root)
