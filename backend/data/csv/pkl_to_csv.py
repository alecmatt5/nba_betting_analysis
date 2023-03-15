import pandas as pd

path_x  = '../pkl/ADV_OHE_TEAM_ALL'
path_y = '../pkl/y_ADVANCED_TEAM_ALL'
df_x = pd.read_pickle(path_x)
df_y = pd.read_pickle(path_y)
df_x.to_csv('new_collab_X.csv')
df_y.to_csv('new_collab_y.csv')
