# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 13:11:39 2019

@author: sparr
"""
from game import text_game

game = text_game()
a = game.agent

game.run_game(agent=a, num_games=10, num_rounds=20, batch_size=5)


story1.to_csv('story1.csv')
story2.to_csv('story2.csv')
story3.to_csv('story3.csv')
story4.to_csv('story4.csv')
story5.to_csv('story5.csv')
story6.to_csv('story6.csv')
game_scores1.to_csv('game_scores1.csv')
game_scores2.to_csv('game_scores2.csv')
game_scores3.to_csv('game_scores3.csv')
game_scores4.to_csv('game_scores4.csv')
game_scores5.to_csv('game_scores5.csv')
game_scores6.to_csv('game_scores6.csv')


a.model_double.save_weights('transferred_weights_double3.h5')
a.model.save_weights('transferred_weights.h5')
a.model_double.load_weights('transferred_weights.h5')

game.run_game(agent=a, num_games=20, num_rounds=256, batch_size=64, training=True)

story7 = game.story.copy()
game_scores7 = game.end_game_scores.copy()
story7.to_csv('story7.csv')
game_scores7.to_csv('game_scores7.csv')

a.model_double.save_weights('transferred_weights_double4.h5')
a.model.save_weights('transferred_weights.h5')
a.model_double.load_weights('transferred_weights.h5')

game.run_game(agent=a, num_games=10, num_rounds=256, batch_size=64, training=True)

story8 = game.story.copy()
game_scores8 = game.end_game_scores.copy()
story8.to_csv('story8.csv')
game_scores8.to_csv('game_scores8.csv')

a.model_double.save_weights('transferred_weights_double5.h5')
a.model.save_weights('transferred_weights.h5')
a.model_double.load_weights('transferred_weights.h5')

game.run_game(agent=a, num_games=10, num_rounds=256, batch_size=64, training=True)

story9 = game.story.copy()
game_scores9 = game.end_game_scores.copy()
story9.to_csv('story9.csv')
game_scores9.to_csv('game_scores9.csv')


game.run_game(agent=a, num_games=10, num_rounds=256, batch_size=64, training=True)
story9 = game.story.copy()
game_scores9 = game.end_game_scores.copy()
story9.to_csv('story9.csv')
game_scores9.to_csv('game_scores9.csv')
a.model_double.save_weights('transferred_weights_double6.h5')
a.model.save_weights('transferred_weights.h5')
a.model_double.load_weights('transferred_weights.h5')
game.run_game(agent=a, num_games=20, num_rounds=256, batch_size=64, training=True)
story11 = game.story.copy()
game_scores11 = game.end_game_scores.copy()
story11.to_csv('story11.csv')
game_scores11.to_csv('game_scores11.csv')
a.model_double.save_weights('transferred_weights_double7.h5')

