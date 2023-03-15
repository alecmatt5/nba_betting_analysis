import streamlit as st
from streamlit_option_menu import option_menu

import pandas as pd
import numpy as np

import datetime
import requests

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

# get colors from theme config file, or set the colours to altair standards
# if chck:
#     primary_clr = st.get_option("theme.primaryColor")
#     txt_clr = st.get_option("theme.textColor")
#     # I want 3 colours to graph, so this is a red that matches the theme:
#     second_clr = "#d87c7c"
# else:
#     primary_clr = '#4c78a8'
#     second_clr = '#f58517'
#     txt_clr = '#e45756'

selected = option_menu(
    menu_title=None,
    options=['Yesterday', 'Today', 'Tommorow'],
    default_index=1,
    icons=['body-text','calendar3','calculator'],
    orientation="horizontal"
)

games = pd.read_pickle('../backend/data/pkl/sbr_current_betting_data_2023-03-09.pkl')


if selected == 'Yesterday':
    df = pd.DataFrame(
    np.random.randn(10, 5),
    columns=('col %d' % i for i in range(5)))

    st.table(df)

    option = st.selectbox(
        'How would you like to be contacted?',
        ('Email', 'Home phone', 'Mobile phone'))

    st.write('You selected:', option)

if selected == 'Today':

    show_Betmgm = st.checkbox('Betmgm', value=True)
    show_Draft_Kings = st.checkbox('Draft_Kings', value=True)
    show_Fanduel = st.checkbox('Fanduel', value=True)
    show_Caesars = st.checkbox('Caesars', value=True)
    show_Pointsbet = st.checkbox('Pointsbet', value=True)
    show_Wynn = st.checkbox('Wynn', value=True)
    show_Betrivers = st.checkbox('Betrivers', value=True)


    def display_dataframe(show_Betmgm, show_Draft_Kings, show_Fanduel, show_Caesars, show_Pointsbet, show_Wynn, show_Betrivers):
        # Hide the columns that are not selected
        df = games

        if not show_Betmgm:
            df.drop(['Betmgm_Spread', 'Betmgm_Odds'], axis=1, inplace=True)
        if not show_Draft_Kings:
            df.drop(['Draft_Kings_Spread', 'Draft_Kings_Odds'], axis=1, inplace=True)
        if not show_Fanduel:
            df.drop(['Fanduel_Spread', 'Fanduel_Odds'], axis=1, inplace=True)
        if not show_Caesars:
            df.drop(['Caesars_Spread', 'Caesars_Odds'], axis=1, inplace=True)
        if not show_Pointsbet:
            df.drop(['Pointsbet_Spread', 'Pointsbet_Odds'], axis=1, inplace=True)
        if not show_Wynn:
            df.drop(['Wynn_Spread', 'Wynn_Odds'], axis=1, inplace=True)
        if not show_Betrivers:
            df.drop(['Betrivers_Spread', 'Betrivers_Odds'], axis=1, inplace=True)

        # Display the data frame with the selected columns using streamlit
        st.write(df)

    # def display_dataframe(show_Betmgm, show_Draft_Kings, show_Fanduel, show_Caesars, show_Pointsbet, show_Wynn, show_Betrivers):
    #     # Hide the columns that are not selected
    #     df = df = pd.DataFrame()

    #     if not show_Betmgm:
    #         df.join(games[['Betmgm_Spread', 'Betmgm_Odds']])
    #     if not show_Draft_Kings:
    #         df.join(games[['Draft_Kings_Spread', 'Draft_Kings_Odds']])
    #     if not show_Fanduel:
    #         df.join(games[['Fanduel_Spread', 'Fanduel_Odds']])
    #     if not show_Caesars:
    #         df.join(games[['Caesars_Spread', 'Caesars_Odds']])
    #     if not show_Pointsbet:
    #         df.join(games[['Pointsbet_Spread', 'Pointsbet_Odds']])
    #     if not show_Wynn:
    #         df.join(games[['Wynn_Spread', 'Wynn_Odds']])
    #     if not show_Betrivers:
    #         df.join(games[['Betrivers_Spread', 'Betrivers_Odds']])

    #     # Display the data frame with the selected columns using streamlit
    #     st.write(df)

    # Call the display_dataframe function with the current states of the checkboxes as arguments
    display_dataframe(show_Betmgm, show_Draft_Kings, show_Fanduel, show_Caesars, show_Pointsbet, show_Wynn, show_Betrivers)

    st.dataframe(games.style.highlight_max(subset=['Opening_Spread', 'Betmgm_Spread',
                                                   'Draft_Kings_Spread', 'Fanduel_Spread',
                                                   'Caesars_Spread', 'Pointsbet_Spread',
                                                   'Wynn_Spread', 'Betrivers_Spread',],
                                           axis=1, color='grey').highlight_min(subset=['Opening_Odds', 'Betmgm_Odds',
                                                   'Draft_Kings_Odds', 'Fanduel_Odds',
                                                   'Caesars_Odds', 'Pointsbet_Odds',
                                                   'Wynn_Odds', 'Betrivers_Odds'],
                                           axis=1, color='brown'))

    button_1 = st.button('Today')
    button_2 = st.button('Tommorow')

    # Define a function to display the data frame based on the button clicked
    def display_dataframe(button_1, button_2):
        if button_1:
            st.write(games[['Opening_Spread', 'Betmgm_Spread']])
        elif button_2:
            st.write(games[['Draft_Kings_Spread', 'Fanduel_Spread']])

    # Call the display_dataframe function with the button states as arguments
    display_dataframe(button_1, button_2)


if selected == 'Tommorow':
    '''
    # TaxiFareModel front

    This front queries the Le Wagon [taxi fare model API](https://taxifare.lewagon.ai/predict?pickup_datetime=2012-10-06%2012:10:20&pickup_longitude=40.7614327&pickup_latitude=-73.9798156&dropoff_longitude=40.6513111&dropoff_latitude=-73.8803331&passenger_count=2)
    '''

    with st.form(key='params_for_api'):

        pickup_date = st.date_input('pickup datetime', value=datetime.datetime(2012, 10, 6, 12, 10, 20))
        pickup_time = st.time_input('pickup datetime', value=datetime.datetime(2012, 10, 6, 12, 10, 20))
        pickup_datetime = f'{pickup_date} {pickup_time}'
        pickup_longitude = st.number_input('pickup longitude', value=40.7614327)
        pickup_latitude = st.number_input('pickup latitude', value=-73.9798156)
        dropoff_longitude = st.number_input('dropoff longitude', value=40.6413111)
        dropoff_latitude = st.number_input('dropoff latitude', value=-73.7803331)
        passenger_count = st.number_input('passenger_count', min_value=1, max_value=8, step=1, value=1)

        st.form_submit_button('Make prediction')

    params = dict(
        pickup_datetime=pickup_datetime,
        pickup_longitude=pickup_longitude,
        pickup_latitude=pickup_latitude,
        dropoff_longitude=dropoff_longitude,
        dropoff_latitude=dropoff_latitude,
        passenger_count=passenger_count)

    wagon_cab_api_url = 'https://taxifare.lewagon.ai/predict'
    response = requests.get(wagon_cab_api_url, params=params)

    prediction = response.json()

    pred = prediction['fare']

    st.header(f'Fare amount: ${round(pred, 2)}')
