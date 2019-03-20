# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 17:38:19 2019

@author: sparr
"""

from game import text_game
import cProfile
import pstats

game = text_game()
a = game.agent

pr = cProfile.Profile()
pr.enable()
game.run_game(agent=a, num_games=1, num_rounds=5, batch_size=50)
pr.disable()
ps = pstats.Stats(pr).sort_stats('time')
ps.print_stats(30)



