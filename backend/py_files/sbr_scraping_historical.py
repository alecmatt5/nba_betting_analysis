from bs4 import BeautifulSoup
import requests
import pandas as pd
import time
from py_files.preprocess import get_basic_boxscores

games = get_basic_boxscores(date='2023-03-01')
games['GAME_DATE'] = games['GAME_DATE'].dt.strftime('%Y-%m-%d')
games = games.sort_values('GAME_DATE', ascending=False)
betting_data = pd.read_pickle('data/pkl/sbr_betting_data.pkl')
last_date = betting_data ['Game_Date'].iloc[0]
dates = games[games['GAME_DATE'] > last_date].sort_values('GAME_DATE', ascending=False)['GAME_DATE'].unique().tolist()


first = True
for date in dates:
    # time.sleep(4)
    URL = f"https://www.sportsbookreview.com/betting-odds/nba-basketball/?date={date}"
    headers = {"User-Agent" : "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:99.0) Gecko/20100101 Firefox/99.0"}

    page = requests.get(url=URL) #headers=headers)
    soup = BeautifulSoup(page.content, 'html.parser')

    teams = soup.find_all(class_='GameRows_participantBox__0WCRz')
    spread = soup.find_all(class_="OddsCells_compact__cawia border-left")
    wager_percentage = soup.find_all("span", class_="opener")
    scores = soup.find_all(class_ = 'GameRows_scores__YkN24')
    opening_spread = soup.find_all(class_='GameRows_adjust__NZn2m GameRows_opener__NivKJ')

    temp = []
    for j in range(int(7*len(teams)/2)):
        for w in [1, 2, 4, 5]:
            temp.append(spread[j].find_all("span")[w].text)

    spreads = []
    odds = []
    for i in range(len(temp)):
        if i % 2 == 0:
            spreads.append(temp[i])
        else:
            odds.append(temp[i])

    away = []
    home = []
    for i in range(len(spreads)):
        if i % 2 == 0:
            away.append(spreads[i])
            away.append(odds[i])
        else:
            home.append(spreads[i])
            home.append(odds[i])

    rows_to_add= []
    temp_away = []
    temp_home = []
    splits = [i for i in range(13, (14*20), 14)]
    for i in range(len(away)):
        temp_away.append(away[i])
        temp_home.append(home[i])
        if i in splits:
            rows_to_add.append(temp_away)
            rows_to_add.append(temp_home)
            temp_away = []
            temp_home = []

    row = []
    for i in range(len(teams)):
        if i % 2 == 0:
            opponent = teams[i+1].text
            home = 0
        else:
            opponent = teams[i-1].text
            home = 1
        row.append([date, teams[i].text, home, opponent, scores[i].text, wager_percentage[i].text, opening_spread[i].text, '-110'])

    df_1 = pd.DataFrame(row, columns=['Game_Date', 'Team_Name', 'Home', 'Opponent', 'Points', 'Pct_of_Bets', 'Opening_Spread', 'Opening_Odds'])

    columns = ['Betmgm_Spread', 'Betmgm_Odds', 'Draft_Kings_Spread', 'Draft_Kings_Odds',
            'Fanduel_Spread', 'Fanduel_Odds', 'Caesars_Spread', 'Caesars_Odds',
            'Pointsbet_Spread', 'Pointsbet_Odds', 'Wynn_Spread', 'Wynn_Odds',
            'Betrivers_Spread', 'Betrivers_Odds']

    df_2 = pd.DataFrame(rows_to_add, columns=columns)

    new_df = df_1.merge(df_2, left_index=True, right_index=True)

    if new_df.shape[0] != 0:
        if first == True:
            betting_df = new_df
            first = False
        else:
            betting_df = pd.concat([betting_df, new_df])
    print(f'{date} scraped')
betting_df = pd.concat([betting_df, betting_data])
betting_df.to_pickle('sbr_betting_data.pkl')
