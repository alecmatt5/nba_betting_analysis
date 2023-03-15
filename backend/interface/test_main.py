from backend.py_files.Ngboost_pipeline import get_y_pred_percentile_from_df
from backend.logic.today_games_preprocessor import preprocess_advanced
from pathlib import Path

preproc_part1, X_features = preprocess_advanced('boxscores_advanced_team_all.pkl',
                                        roll_methods=['mean'],
                                        ohe=True,
                                        scaled=False)
file_path_model = Path.home()/'code'/'alecmatt5'/'nba_betting_analysis'/'backend'/'data'/'pkl'/'ngdemo.pkl'
get_y_pred_percentile_from_df(preproc_part1, file_path_model, new_df=False, percentile=0.54)
