import streamlit as st
from streamlit_option_menu import option_menu

import pandas as pd
import numpy as np

import datetime
import requests

games_yesterday = pd.read_pickle('betting_predictions_2023-03-16.pkl')
games_today = pd.read_pickle('betting_predictions_2023-03-17.pkl')

st.set_page_config(layout="wide",
                   page_title="NBA betting analysis",
                   initial_sidebar_state="expanded",
                   menu_items={'About': None,
                               'Get Help': None,
                               'Report a bug': None})

with st.sidebar:
    # "Use the widgets to alter the graphs:"
    # chck = st.sidebar.checkbox("Use your theme colours on graphs", value=True) # get colours for graphs

    '''
    # PROJECT OVERVIEW

    Introducing a new NBA game prediction website that uses advanced algorithms to predict the outcome of upcoming games and compares them to existing bet spreads. Our technology analyzes various factors, including team performance, player statistics, and game trends, to provide accurate predictions for each game.

    Our site offers a comprehensive comparison of our predictions to existing bet spreads, providing users with the information needed to make informed betting decisions. Our interface is user-friendly and allows for easy navigation of upcoming games, predicted scores, and real-time comparison of predicted scores to bet spreads.

    Whether you are an experienced sports bettor or just starting, our predictions and analysis can provide an edge in making smart betting decisions.

    Join our community of NBA enthusiasts and start predicting today.
    '''

selected = option_menu(
    menu_title=None,
    options=['Yesterday', 'Today'],
    default_index=1,
    icons=['calendar3','calculator'],
    orientation="horizontal"
)


def preprocess(df):
    columns_to_drop = ['Game_Date', 'Home', 'Pct_of_Bets']
    df = df.copy().drop(columns=columns_to_drop)
    df = df.set_index('Team_Name')
    df = df.rename(columns={'y_pred': 'Predictions', 'bet': 'Bet'})
    return df

df_yesterday = preprocess(games_yesterday)
df_today = preprocess(games_today)

outcome = [-19, 19, 17, -17, -5, 5, -16, 16, 0, 0]
results = ["", "Loss", "", "", "", "", "Win", "", "", ""]
df_yesterday["Outcome"] = outcome
df_yesterday["Results"] = results

col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)

with col1:
    show_Betmgm = st.checkbox('Betmgm', value=True)
    show_Draft_Kings = st.checkbox('Draft_Kings', value=False)
with col2:
    show_Fanduel = st.checkbox('Fanduel', value=True)
    show_Caesars = st.checkbox('Caesars', value=False)
with col3:
    show_Pointsbet = st.checkbox('Pointsbet', value=True)
    show_Wynn = st.checkbox('Wynn', value=False)
with col4:
    show_Betrivers = st.checkbox('Betrivers', value=False)
with col5:
    st.empty()
with col6:
    st.empty()
with col7:
    st.empty()
with col8:
    highlight = st.checkbox('Highlight', value=True)


def display_dataframe(df, show_Betmgm, show_Draft_Kings, show_Fanduel, show_Caesars, show_Pointsbet, show_Wynn, show_Betrivers, highlight):

    # spread_subset=['Opening_Spread', 'Betmgm_Spread',
    #         'Draft_Kings_Spread', 'Fanduel_Spread',
    #         'Caesars_Spread', 'Pointsbet_Spread',
    #         'Wynn_Spread', 'Betrivers_Spread',]
    # odds_subset=['Opening_Odds', 'Betmgm_Odds',
    #         'Draft_Kings_Odds', 'Fanduel_Odds',
    #         'Caesars_Odds', 'Pointsbet_Odds',
    #         'Wynn_Odds', 'Betrivers_Odds']
    odds_subset=['Betmgm_Odds',
            'Draft_Kings_Odds', 'Fanduel_Odds',
            'Caesars_Odds', 'Pointsbet_Odds',
            'Wynn_Odds', 'Betrivers_Odds']


    if not show_Betmgm:
        df.drop(['Betmgm_Spread', 'Betmgm_Odds'], axis=1, inplace=True)
        # spread_subset.remove('Betmgm_Spread')
        odds_subset.remove('Betmgm_Odds')
    if not show_Draft_Kings:
        df.drop(['Draft_Kings_Spread', 'Draft_Kings_Odds'], axis=1, inplace=True)
        # spread_subset.remove('Draft_Kings_Spread')
        odds_subset.remove('Draft_Kings_Odds')
    if not show_Fanduel:
        df.drop(['Fanduel_Spread', 'Fanduel_Odds'], axis=1, inplace=True)
        # spread_subset.remove('Fanduel_Spread')
        odds_subset.remove('Fanduel_Odds')
    if not show_Caesars:
        df.drop(['Caesars_Spread', 'Caesars_Odds'], axis=1, inplace=True)
        # spread_subset.remove('Caesars_Spread')
        odds_subset.remove('Caesars_Odds')
    if not show_Pointsbet:
        df.drop(['Pointsbet_Spread', 'Pointsbet_Odds'], axis=1, inplace=True)
        # spread_subset.remove('Pointsbet_Spread')
        odds_subset.remove('Pointsbet_Odds')
    if not show_Wynn:
        df.drop(['Wynn_Spread', 'Wynn_Odds'], axis=1, inplace=True)
        # spread_subset.remove('Wynn_Spread')
        odds_subset.remove('Wynn_Odds')
    if not show_Betrivers:
        df.drop(['Betrivers_Spread', 'Betrivers_Odds'], axis=1, inplace=True)
        # spread_subset.remove('Betrivers_Spread')
        odds_subset.remove('Betrivers_Odds')

    float_cols = df.select_dtypes(include=['float32', 'float64']).columns.tolist()
    format_dict = {}
    for i in float_cols:
        format_dict[i] = "{:.2f}"

    if highlight:
        teams_to_bet_on = df.index[df['Bet'] == True].tolist()
        st.write(df.style.format(format_dict).highlight_max(subset=pd.IndexSlice[teams_to_bet_on, ['Predictions']],
                                        axis=1, color='grey')
                .highlight_max(subset=pd.IndexSlice[teams_to_bet_on, odds_subset],
                                        axis=1, color='brown'), height=36*(df.shape[0]+1))
    else:
        st.write(df.style.format(format_dict), height=36*(df.shape[0]+1))

    # if highlight:
    #     st.dataframe(df.style.highlight_max(subset=spread_subset,
    #                                     axis=1, color='grey')
    #             .highlight_min(subset=odds_subset,
    #                                     axis=1, color='brown'), height=36*(df.shape[0]+1))
    # else:
    #     st.dataframe(df, height=36*(df.shape[0]+1))

if selected == 'Yesterday':
    # Call the display_dataframe function with the current states of the checkboxes as arguments
    display_dataframe(df_yesterday, show_Betmgm, show_Draft_Kings, show_Fanduel, show_Caesars, show_Pointsbet, show_Wynn, show_Betrivers, highlight)

if selected == 'Today':
    # Call the display_dataframe function with the current states of the checkboxes as arguments
    display_dataframe(df_today, show_Betmgm, show_Draft_Kings, show_Fanduel, show_Caesars, show_Pointsbet, show_Wynn, show_Betrivers, highlight)



# button_1 = st.button('Today')
# button_2 = st.button('Tommorow')

# # Define a function to display the data frame based on the button clicked
# def display_dataframe(button_1, button_2):
#     if button_1:
#         st.write(games[['Opening_Spread', 'Betmgm_Spread']])
#     elif button_2:
#         st.write(games[['Draft_Kings_Spread', 'Fanduel_Spread']])

# # Call the display_dataframe function with the button states as arguments
# display_dataframe(button_1, button_2)



# nba_api_url = 'https://nba-betting-analysis-asoblteiuq-uc.a.run.app/predict?percentile_target=0.54'
# response = requests.get(nba_api_url)

# prediction = response.json()

# st.json(prediction)
