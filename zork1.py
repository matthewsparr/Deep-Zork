# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 13:11:39 2019

@author: sparr
"""
from game import text_game

game = text_game()
a = game.agent

game.run_game(agent=a, num_games=10, num_rounds=20, batch_size=5)