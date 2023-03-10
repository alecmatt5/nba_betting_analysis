import streamlit as st
from streamlit_option_menu import option_menu

import pandas as pd
import numpy as np

import datetime
import requests

st.set_page_config(layout="wide",
                   page_title="NBA betting analysis",
                   initial_sidebar_state="expanded",
                   menu_items={'About': "### This is a NBA betting analysis app:)"})

st.sidebar.write("Use the widgets to alter the graphs:")
chck = st.sidebar.checkbox("Use your theme colours on graphs", value=True) # get colours for graphs

# get colors from theme config file, or set the colours to altair standards
if chck:
    primary_clr = st.get_option("theme.primaryColor")
    txt_clr = st.get_option("theme.textColor")
    # I want 3 colours to graph, so this is a red that matches the theme:
    second_clr = "#d87c7c"
else:
    primary_clr = '#4c78a8'
    second_clr = '#f58517'
    txt_clr = '#e45756'


selected = option_menu(
    menu_title=None,
    options=['About', 'Model', ':)'],
    icons=['body-text','calculator','calendar3'],
    orientation="horizontal"
)


if selected == 'About':
    '''
    # NBA

    This front queries the Le Wagon [taxi fare model API](https://taxifare.lewagon.ai/predict?pickup_datetime=2012-10-06%2012:10:20&pickup_longitude=40.7614327&pickup_latitude=-73.9798156&dropoff_longitude=40.6513111&dropoff_latitude=-73.8803331&passenger_count=2)
    '''

if selected == 'Model':
    df = pd.DataFrame(
    np.random.randn(10, 5),
    columns=('col %d' % i for i in range(5)))

    st.table(df)

    option = st.selectbox(
        'How would you like to be contacted?',
        ('Email', 'Home phone', 'Mobile phone'))

    st.write('You selected:', option)

if selected == ':)':
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
