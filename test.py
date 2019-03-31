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
#a.model.load_weights('weights1.h5')
a.epsilon_decay = 1
a.epsilon = 1

#pr = cProfile.Profile()
#pr.enable()
#game.run_game(agent=a, num_games=128, num_rounds=256, batch_size=0, training=False)


#story1 = game.story.copy()
#game_scores1 = game.end_game_scores.copy()

a.epsilon_decay = 0.9995
a.epsilon = 1

game.run_game(agent=a, num_games=5, num_rounds=256, batch_size=64, training=True)

story3 = game.story.copy()
game_scores3 = game.end_game_scores.copy()


a.transfer_learning()

game.run_game(agent=a, num_games=5, num_rounds=256, batch_size=64, training=True)

story4 = game.story.copy()
game_scores4 = game.end_game_scores.copy()

a.transfer_learning()

game.run_game(agent=a, num_games=5, num_rounds=256, batch_size=64, training=True)

story5 = game.story.copy()
game_scores5 = game.end_game_scores.copy()

a.transfer_learning()

game.run_game(agent=a, num_games=5, num_rounds=256, batch_size=64, training=True)

story6 = game.story.copy()
game_scores6 = game.end_game_scores.copy()

#state_data = pd.read_pickle('state_data.pickle')

#story1.plot.line(x='Total_Moves', y='Score')
#game_scores1.plot.line(x='Game Number', y='Score')
#story2.plot.line(x='Total_Moves', y='Score')
#game_scores2.plot.line(x='Game Number', y='Score')

