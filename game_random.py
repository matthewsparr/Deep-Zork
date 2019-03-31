# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 09:16:37 2019

@author: sparr
"""


# coding: utf-8

# In[62]:
from queue import Queue, Empty
from threading  import Thread
from subprocess import Popen, PIPE
from spacy.matcher import Matcher
from spacy.attrs import POS
from itertools import permutations
from numpy.random import choice
import pickle
import re
from progressbar import ProgressBar
import numpy as np
import pandas as pd
from time import sleep
import traceback


class text_game_random:
    
    def __init__(self):
        from game_commands import commands
        import spacy
        import re
        
        self.emulator_file = 'dfrotz.exe'
        self.game_file = 'zork1.z5'
        
        self.p = None
        self.q = None
        self.t = None
        
        self.compiled_expression = re.compile('[^ \-\sA-Za-z0-9"\']+')
        
        self.nlp = spacy.load('en_core_web_sm')

        self.sleep_time = 0.01

        self.score = 0
        self.game_score = 0
        self.game_score_weight = 1
        self.negative_per_turn_reward = 1
        self.inventory_reward_value = 3
        self.new_area_reward_value = 2
        self.moving_around_reward_value = 0.5
        self.inventory_not_new_reward_value = 0.5
                
        cmds = commands()
        self.basic_actions = cmds.basic_actions
        self.directions = cmds.directions
        self.command1_actions = cmds.command1_actions
        self.command2_actions = cmds.command2_actions
        self.action_space = cmds.action_space
        self.filtered_tokens = cmds.filtered_tokens
        self.invalid_nouns = [] 
        self.valid_nouns = []
        
        self.unique_state = set()
        self.actions_probs_dict = dict()
        self.story_random = pd.DataFrame(columns=['Surroundings', 'Inventory', 'Action', 'Response', 'Reward', 'Reward_Type', 'Score', 'Moves', 'Total_Moves'])
        self.end_game_scores_random = pd.DataFrame(columns=['Game Number', 'Score'])
        self.stories_random = []

        self.state_data_random = pd.DataFrame(columns=['State', 'Actions'])
        
    def enqueue_output(self, out, queue):
        for line in iter(out.readline, b''):
            queue.put(line)
        out.close()

    def start_game(self):
        self.story_random = pd.DataFrame(columns=['Surroundings', 'Inventory', 'Action', 'Response', 'Reward', 'Reward_Type', 'Score', 'Moves', 'Total_Moves'])
        self.end_game_scores_random = pd.DataFrame(columns=['Game Number', 'Score'])
        self.score = 0
        self.unique_state = set()
        self.game_score = 0
        self.p = Popen([self.emulator_file, self.game_file], stdin=PIPE, stdout=PIPE, stderr=PIPE, shell=True)
        self.q = Queue()
        self.t = Thread(target=self.enqueue_output, args=(self.p.stdout, self.q))
        self.t.daemon = True # thread dies with the program
        self.t.start()
        sleep(self.sleep_time*10)
        return()
    
    def end_game(self):
        self.kill_game()
        
    def kill_game(self):
        self.p.terminate()
        self.p.kill()
        
    def restart_game(self):
        self.perform_action('restart')
        self.readLine()
        self.perform_action('y')
        self.readLine()
        self.score = 0
        self.unique_state = set()
        self.game_score = 0
        
    # read line without blocking
    def readLine(self):
        cont = True
        state = ""
        while cont:
            try:  line = self.q.get_nowait() # or q.get(timeout=.1)
            except Empty:
                cont = False
            else: 
                state = state + line.decode("utf-8").replace('\n', " ").replace('\r', " ")
        if ('840726' in state): ## Opening state
            state = state[state.index('840726') + len('840726'):]
        try:
            score, moves = self.grab_score_moves(state)
            state = state[state.index('Moves: ')+len('Moves:')+5:-1].strip()
        except:  ## not valid move
            pass
        sleep(self.sleep_time)
        return(state, score, moves)
        
    def get_state(self):
        ## check surroundings
        self.perform_action('look')
        surroundings,score,moves = self.readLine()
        surroundings = self.preprocess(surroundings)
        ## check inventory
        self.perform_action('inventory')
        inventory,score,moves = self.readLine()
        inventory = self.preprocess(inventory)
        
        ## join surroundings and inventory
        state = surroundings + ' ' + inventory
        return state, surroundings, inventory, score, moves
    
    def grab_score_moves(self, state):
        try:
            score = int(state[state.index('Score: ') + len('Score: '):][0:3].strip())
            moves = int(state[state.index('Moves: ') + len('Moves: '):][0:3].strip())
        except:  ## not valid move
            score = 0
            moves = 0
        return(score, moves)

    def look_surroundings(self, p):
         self.perform_action('look')

    def check_inventory(self, p):
         self.perform_action('inventory')

    def get_nouns(self, state):
        matcher = Matcher(self.nlp.vocab)
        matcher.add('Noun phrase', None, [{POS: 'NOUN'}])
        doc = self.nlp(state)
        matches = matcher(doc)
        noun_set = set()
        for id, start, end in matches:
            noun = doc[start:end].text
            if noun not in self.directions and noun not in self.invalid_nouns:
                noun_set.add(noun)
        return(noun_set)
        
    def generate_action_tuples(self, nouns):
        possible_actions = []
        for i in self.basic_actions:
            possible_actions.append(i)
        for i in nouns:
            for action1 in self.command1_actions:   ## first loop replaces 'x' in each action in command1_actions
                action_to_add = action1.replace('OBJ', i)
                possible_actions.append(action_to_add)
            noun_permutations = list(permutations(nouns, 2))    ## second loop replaces 'x' and 'y' in each action in command2_actions
            for action2 in self.command2_actions:
                for perm in noun_permutations:
                    if (perm[0] == perm[1]):  ## ignore same noun acting on itself
                        pass
                    else:
                        possible_actions.append(action_to_add)
        return possible_actions
    
    def select_one(self, action_space, similarities):
        random_index = choice(len(action_space))
        return action_space[random_index]
    
    def add_to_action_space(self, action_space, actions):
        for action in actions:
            action_space.add(action)
        return action_space
        
    def perform_action(self, command):
        self.p.stdin.write(bytes(command+ "\n", 'ascii'))
        self.p.stdin.flush()
        sleep(self.sleep_time)## wait for action to register
        
    def preprocess(self, text):
        # fix bad newlines (replace with spaces), unify quotes
        text = text.strip()
        text = text.replace('\\n', '').replace('‘', '\'').replace('’', '\'').replace('”', '"').replace('“', '"')
        # convert to lowercase
        text = text.lower()
        # remove all characters except alphanum, spaces and - ' "
        text = self.compiled_expression.sub('', text)
        # split numbers into digits to avoid infinite vocabulary size if random numbers are present:
        #text = re.sub('[0-9]', ' \g<0> ', text)
        
        ## remove problematic flavor text
        flavor_text = 'you hear in the distance the chirping of a song bird'
        if (flavor_text in text):
            text = text.replace(flavor_text, '')
        text = re.sub('\s{2,}', ' ', text)
        # expand unambiguous 'm, 't, 're, ... expressions
        #text = text.replace('\'m ', ' am ').replace('\'re ', ' are ').replace('won\'t', 'will not').replace('n\'t', ' not').replace('\'ll ', ' will ').replace('\'ve ', ' have ').replace('\'s', ' \'s')
        return text

    def calculate_reward(self, inventory, old_inventory, moves_count, old_state, new_state, round_score):
        reward = 0
        reward_msg = ''
        ## add reward from score in game
        if(moves_count != 0):
            reward = reward + round_score*self.game_score_weight
            if (round_score > 0):
                print('Scored ' + str(round_score) + ' points in game.')
                reward_msg += ' game score: ' + str(round_score) + ' '
        ## add small negative reward for each move
        reward = reward - self.negative_per_turn_reward
        
        ## add reward for picking up / using items
        if(moves_count != 0):
            if inventory.strip().lower() not in old_inventory.strip().lower(): ## inventory changed, ignoring chirping bird line
                ## keep track of unique inventory changes to prevent picking up and dropping items constantly
                if (old_inventory + ' - ' + inventory) not in self.unique_inventory_changes:
                    self.unique_inventory_changes.add(old_inventory + ' - ' + inventory)
                    reward = reward + self.inventory_reward_value
                    print('inventory changed - new')
                    reward_msg += ' inventory score (' + old_inventory + " --- " + inventory + ')'
                else:
                    reward = reward + self.inventory_not_new_reward_value
                    
        ## add reward for discovering new areas
        if new_state.strip() not in self.unique_state:  ## new location
            reward = reward + self.new_area_reward_value
            self.unique_state.add(new_state.strip())
            reward_msg += ' new area score ---' + new_state.strip()
        
        if old_state not in new_state:
            reward = reward + self.moving_around_reward_value
            reward_msg += ' - moved around - ' 

        print('Rewarded: ' + str(reward) + ' points.')
        return reward, reward_msg
        
    def get_data(self, state):
        ## if we have generated actions before for state, load them, otherwise generate actions
        if (state in list(self.state_data_random['State'])):
            actions = list(self.state_data_random[self.state_data['State'] == state]['Actions'])[0]
        else: 
            ## get nouns from state
            nouns = self.get_nouns(state)

            # build action space
            current_action_space = self.generate_action_tuples(nouns)
            action_space = set()
            action_space = self.add_to_action_space(action_space, current_action_space)
            actions = []
            for a in action_space:
                actions.append(a)

            ## store state data 
            row = len(self.state_data_random)
            self.state_data_random.loc[row, 'State'] = state
            self.state_data_random.loc[row, 'Actions'] = [actions]
        return actions
    
    def perform_selected_action(self, action):
        self.perform_action(action)
        response,current_score,moves = self.readLine()
        response = self.preprocess(response)
        return response, current_score, moves

    def run_game(self, agent, num_games, num_rounds):
        
        ## initialize progress bar
        pbar = ProgressBar(maxval=num_rounds*num_games)
        pbar.start()
        
        ## initialize game
        self.start_game()

        ## number of games loop
        for game_number in range(0, num_games):
            print('___starting new game___')
            new_state = ''
            inventory = ''
            try:
                ## number of rounds loop
                for i in range(0, num_rounds):
                    
                    ## get initial state if first round, else grab new state from previous round
                    if (i==0):
                        state, old_surroundings, old_inventory, _, _, = self.get_state()
                        self.unique_state.add(state)
                    else:
                        state  = new_state
                        old_inventory = inventory
                        
                    ## sometimes reading of lines gets backed up, if this happens reset and re-check state
                    invalid_line = True
                    while invalid_line:
                        invalid_line = False
                        if len(state) > self.state_limit or len(state)<5 or 'score' in state:
                            print('encountered line read bug')
                            state, old_surroundings, old_inventory, _, _, = self.get_state()
                            invalid_line = True
                    print(state)

                        
                    ## get data for current state
                    actions = self.get_data(state)
                    

                    action = self.select_one(actions)       
                    print('-- ' + action + ' --')
    
                    ## perform selected action
                    response, current_score, moves = self.perform_selected_action(action)
                    
                    ## check new state after performing action
                    new_state, surroundings, inventory, current_score, moves = self.get_state()
                    new_state = self.preprocess(new_state)
                    
                    ## get reward
                    round_score = current_score - self.game_score
                    self.game_score = current_score
                    reward, reward_msg = self.calculate_reward(inventory, old_inventory, i, state, new_state, round_score)
    
                    ## update story dataframe
                    self.score += reward
                    total_round_number = i + game_number*num_rounds
                    self.story_random.loc[total_round_number] = [state, old_inventory, action, response, reward, 
                                  reward_msg, self.score, str(i), total_round_number]
                    
                    ## update progress bar
                    pbar.update(i + (game_number)*num_rounds) 
                    
                self.end_game_scores_random.loc[game_number] = [game_number, self.score]
                self.restart_game()
            except Exception as e:
                print('exception')
                traceback.print_tb(e.__traceback__)
                self.restart_game()
                #print(e.with_traceback())
            pbar.finish()
            self.stories.append(self.story)
        self.state_data.to_pickle('state_data_random.pickle')
        self.kill_game()
        return True

