# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 17:38:19 2019

@author: sparr
"""

from game import text_game
import cProfile
import pstats
import pandas as pd

game = text_game()
a = game.agent
a.epsilon_decay = 1

pr = cProfile.Profile()
pr.enable()
game.run_game(agent=a, num_games=20, num_rounds=256, batch_size=64)
pr.disable()
ps = pstats.Stats(pr).sort_stats('time')

story = game.story.copy()
game_scores = game.end_game_scores.copy()
a.epsilon = 0.00001
game.run_game(agent=a, num_games=1, num_rounds=64, batch_size=64)
story2 = game.story.copy()
game_scores2 = game.end_game_scores.copy()

state_data = pd.read_pickle('state_data.pickle')

story.plot.line(x='Total_Moves', y='Score')
game_scores.plot.line(x='Game Number', y='Score')

story2.plot.line(x='Total_Moves', y='Score')
game_scores2.plot.line(x='Game Number', y='Score')

ps.print_stats(30)

