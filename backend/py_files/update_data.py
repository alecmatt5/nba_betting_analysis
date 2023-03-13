import pandas as pd
import numpy as np
import os
import requests
from nba_api.stats.static import teams
from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.endpoints import boxscoreadvancedv2
from datetime import datetime, timedelta

def get_game_ids():
    #Game ids for the missing games
    #UPDATE THIS!!!!!!!!!!!!!
    nba_teams = teams.get_teams()
    team_names = [team['full_name'] for team in nba_teams]
    team_names.sort()
    team_ids = [team['id'] for team in nba_teams]

    games = None
    for ids in team_ids:
        if games is None:
            gamefinder = leaguegamefinder.LeagueGameFinder(team_id_nullable=ids)
            games = gamefinder.get_data_frames()[0]
        else:
            gamefinder = leaguegamefinder.LeagueGameFinder(team_id_nullable=ids)
            games = pd.concat([games, gamefinder.get_data_frames()[0]])
    games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'])

    yesterday = (datetime.utcnow() - timedelta(hours=33)).strftime('%Y-%m-%d')

    games= games[games['GAME_DATE'] == yesterday]
    game_ids = games['GAME_ID'].unique().tolist()

    return game_ids

def update_elo():
    today = (datetime.utcnow() - timedelta(hours=9)).strftime('%Y-%m-%d')
    URL = 'https://projects.fivethirtyeight.com/nba-model/nba_elo.csv'
    elo_past = pd.read_csv(URL)
    elo_past['date'] = pd.to_datetime(elo_past['date'])
    elo_past= elo_past[elo_past['date'] > '2018-09-01']
    elo_past[elo_past['date'] < today]
    map = {'BRK': 'BKN', 'CHO': 'CHA', 'PHO': 'PHX'}
    elo_past = elo_past.replace({'team1': map, 'team2': map})
    elo_past = elo_past[['date', 'team1', 'team2', 'elo1_pre', 'elo2_pre', 'raptor1_pre', 'raptor2_pre']]
    elo_past.to_pickle('data/pkl/elo_past.pkl')
    return

def update_raw_advanced():
    #Get existing data

    adv_team = pd.read_pickle('data/pkl/boxscores_advanced_team_all.pkl')
    adv_player = pd.read_pickle('data/pkl/boxscores_advanced_player_all.pkl')

    game_ids = get_game_ids()

    boxscores_advanced_player = None
    boxscores_advanced_team = None
    for game_id in game_ids:
        if boxscores_advanced_team is None:
            gamefinder = boxscoreadvancedv2.BoxScoreAdvancedV2(game_id=game_id)
            boxscores_advanced_team = gamefinder.get_data_frames()[1]
            boxscores_advanced_player = gamefinder.get_data_frames()[0]
        else:
            gamefinder = boxscoreadvancedv2.BoxScoreAdvancedV2(game_id=game_id)
            boxscores_advanced_team = pd.concat([boxscores_advanced_team, gamefinder.get_data_frames()[1]])
            boxscores_advanced_player = pd.concat([boxscores_advanced_player, gamefinder.get_data_frames()[0]])

    #add new rows to existing data frame
    adv_team = pd.concat([adv_team, boxscores_advanced_team])
    adv_player = pd.concat([adv_player, boxscores_advanced_player])

    #update the pickle file with all the data
    adv_team.to_pickle('data/pkl/boxscores_advanced_team_all.pkl')
    adv_player.to_pickle('data/pkl/boxscores_advanced_player_all.pkl')

    return

# update_elo()
# update_raw_advanced()
