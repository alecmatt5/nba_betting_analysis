from py_files.ngboost_team import get_y_pred_percentile_api
from py_files.ngboost_team import load_ngboost_team_model
from py_files.today_games_preprocessor import preprocess_advanced
from pathlib import Path
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

today_date = datetime.today().strftime('%Y-%m-%d')
app = FastAPI()
file_path_model = 'data/pkl/ngdemo.pkl'
app.state.model = load_ngboost_team_model(file_path_model)
app.state.df_app = pd.read_pickle(f'data/pkl/demo_{today_date}.pkl')
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
@app.get("/predict")
def predict(percentile_target=0.54):
    model = app.state.model
    y_pred = get_y_pred_percentile_api(app.state.df_app, model,
                                            percentile=percentile_target)
    y_json = y_pred.to_dict()
    return y_json


@app.get("/")
def root():
    X_json = app.state.df_app.to_dict()
    return X_json

if __name__=='__main__':
    y_json = predict(percentile_target=0.54)
    print(y_json)
