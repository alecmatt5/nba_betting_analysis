from logic.ngboost_team import get_y_pred_percentile_api
from logic.ngboost_team import load_ngboost_team_model
from logic.today_games_preprocessor import preprocess_advanced
from pathlib import Path
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
file_path_model = 'data'/'pkl'/'ngdemo.pkl'
app.state.model = load_ngboost_team_model(file_path_model)
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

    preproc_part1, X_features = preprocess_advanced('boxscores_advanced_team_all.pkl',
                                        roll_methods=['mean', 'median', 'std'],
                                        ohe=True,
                                        scaled=False)
    model = app.state.model
    y_pred = get_y_pred_percentile_api(preproc_part1, model,
                                            percentile=percentile_target)
    y_json = y_pred.to_dict()
    return y_json


@app.get("/")
def root():
    # $CHA_BEGIN
    return dict(greeting="Hello")
    # $CHA_END
