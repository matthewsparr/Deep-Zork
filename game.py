
# coding: utf-8

# In[62]:
from queue import Queue, Empty
from threading  import Thread
from subprocess import Popen, PIPE
from spacy.matcher import Matcher
from spacy.attrs import POS
from nltk.tokenize import word_tokenize
from itertools import permutations
from numpy.random import choice
from keras.preprocessing.sequence import pad_sequences
import pickle
import re
from gensim.models import Word2Vec
from progressbar import ProgressBar
import numpy as np
from keras.preprocessing.text import Tokenizer
import pandas as pd
from time import sleep
import traceback
import json


class text_game:
    
    def __init__(self):
        from agent import DQNAgent
        from game_commands import commands
        import spacy
        import re
        
        self.emulator_file = 'dfrotz.exe'
        self.game_file = 'zork1.z5'
        
        self.agent = DQNAgent()
        self.batch_size=5
        
        self.p = None
        self.q = None
        self.t = None
        
        self.compiled_expression = re.compile('[^ \-\sA-Za-z0-9"\']+')
        
        self.tutorials_text = 'tutorials_2.txt'
        self.word_2_vec = self.init_word2vec()
        
        self.nlp = spacy.load('en_core_web_sm')
        
        self.tokenizer = None
        self.vocab_size = 1200
        self.state_limit = 1000
        
        self.sleep_time = 0.05
        
        self.random_action_weight = 6
        self.random_action_basic_prob = 0.5
        self.random_action_low_prob = 0.2
        
        self.score = 0
        self.game_score = 0
        self.game_score_weight = 10
        self.negative_per_turn_reward = 1
        self.inventory_reward_value = 50
        self.new_area_reward_value = 20
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
        
        self.unique_state = set()
        self.actions_probs_dict = dict()
        self.story = pd.DataFrame(columns=['Surroundings', 'Inventory', 'Action', 'Response', 'Reward', 'Reward_Type', 'Score', 'Moves', 'Total_Moves'])
        self.end_game_scores = pd.DataFrame(columns=['Game Number', 'Score'])
        self.stories = []
        
        self.load_invalid_nouns()
        self.init_word2vec()
        self.init_tokenizer()
        
        self.unique_inventory_changes = set()
        self.state_data = pd.DataFrame(columns=['State', 'StateVector', 'ActionData'])
        
    def load_state_data(self):
        try:
            self.state_data = pd.read_pickle('state_data.pickle')
        except:
            self.state_data = pd.DataFrame(columns=['State', 'StateVector', 'ActionData'])
    def enqueue_output(self, out, queue):
        for line in iter(out.readline, b''):
            queue.put(line)
        out.close()

    def start_game(self):
        self.load_state_data()
        self.story = pd.DataFrame(columns=['Surroundings', 'Inventory', 'Action', 'Response', 'Reward', 'Reward_Type', 'Score', 'Moves', 'Total_Moves'])
        self.end_game_scores = pd.DataFrame(columns=['Game Number', 'Score'])
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
        self.save_invalid_nouns()
        self.save_tokenizer()
        self.kill_game()
        
    def kill_game(self):
        self.save_invalid_nouns()
        self.p.terminate()
        self.p.kill()
        
    def restart_game(self):
        self.save_invalid_nouns()
        self.perform_action('restart')
        self.readLine()
        self.perform_action('y')
        self.readLine()
        self.score = 0
        self.unique_state = set()
        self.unique_inventory_changes = set()
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
        print(surroundings)
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
        noun_list = []
        for id, start, end in matches:
            noun = doc[start:end].text
            if noun not in self.directions and noun not in self.invalid_nouns:
                noun_list.append(noun)
        return(noun_list)
        
    def generate_action_tuples(self, nouns):
        possible_actions = []
        similarities = []
        for i in self.basic_actions:
            possible_actions.append(i)
            similarities.append(self.random_action_basic_prob)
        for i in nouns:
            for action1 in self.command1_actions:   ## first loop replaces 'x' in each action in command1_actions
                action_to_add = action1.replace('OBJ', i)
                possible_actions.append(action_to_add)
                try:
                    similarities.append(self.word_2_vec.similarity(word_tokenize(action_to_add)[0], i))
                except:
                    similarities.append(self.random_action_low_prob)
            noun_permutations = list(permutations(nouns, 2))    ## second loop replaces 'x' and 'y' in each action in command2_actions
            for action2 in self.command2_actions:
                for perm in noun_permutations:
                    if (perm[0] == perm[1]):  ## ignore same noun acting on itself
                        pass
                    else:
                        action_to_add = action2.replace('OBJ', perm[0])
                        action_to_add = action_to_add.replace('DCT', perm[1])
                        possible_actions.append(action_to_add)
                        try:
                            similarities.append(self.word_2_vec.similarity(word_tokenize(action_to_add.strip(i))[0], i).mean())
                        except:
                            similarities.append(self.random_action_low_prob)

        return possible_actions
    
    def select_one(self, action_space, similarities):
        random_index = choice(len(action_space), p=similarities)
        return action_space[random_index]
    
    def add_to_action_space(self, action_space, actions):
        similarities = []

        for action in actions:
            action_space.add(action)
        for action in action_space:
            words = word_tokenize(action)
            verb = words[0]
            if verb in self.basic_actions:    ## basic commands i.e. go north, go south
                similarities.append(self.random_action_basic_prob)
            elif len(words)<3:           ## commands with one noun i.e. open mailbox, read letter
                noun = word_tokenize(action)[1]
                try:
                    sim_score = self.word_2_vec.similarity(verb, noun)**self.random_action_weight
                    if sim_score < 0:
                        sim_score = self.random_action_basic_prob**self.random_action_weight
                    similarities.append(sim_score)
                except:
                    similarities.append(self.random_action_low_prob**self.random_action_weight)

            else:                       ## commands with two nouns i.e. unlock chest with key
                try:
                    noun1 = word_tokenize(action)[1]
                    prep = word_tokenize(action)[2]
                    noun2 = word_tokenize(action)[3]
                    sim_score1 = self.word_2_vec.similarity(verb, noun1)
                    sim_score2 = self.word_2_vec.similarity(prep, noun2)
                    sim_score = ((sim_score1 + sim_score2)/2)**self.random_action_weight
                    if sim_score < 0:
                        sim_score = 0.05
                    similarities.append(sim_score**self.random_action_weight)
                except:
                    similarities.append(self.random_action_low_prob**self.random_action_weight)


        return action_space, similarities
        
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
    def vectorize_text(self, text, tokenizer):
        words = word_tokenize(text)
        tokenizer.fit_on_texts(words)
        seq = tokenizer.texts_to_sequences(words)
        sent = []
        for i in seq:
            sent.append(i[0])
        padded = pad_sequences([sent], maxlen=50, padding='post')
        return (padded)
    
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
            if  inventory.strip().lower() not in old_inventory.strip().lower(): ## inventory changed, ignoring chirping bird line
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

    def detect_invalid_nouns(self, action_response):
        word = ''
        ## detect and remove invalid nouns from future turns
        if('know the word' in action_response):
            startIndex = action_response.find('\"')
            endIndex = action_response.find('\"', startIndex + 1)
            word = action_response[startIndex+1:endIndex]
        return word
            
    def save_tokenizer(self):
        ## save invalid nouns to pickled list
        try:
            with open('tokenizer.pickle', 'wb') as fp:
                pickle.dump(self.tokenizer, fp)
        except:
            pass
    
    def load_tokenizer(self):
        ## load previously found invalid nouns from pickled list
        try:
            with open ('tokenizer.pickle', 'rb') as fp:
                n = pickle.load(fp)
                self.tokenizer.extend(n)
        except:
            pass        
            
    def save_invalid_nouns(self):
        ## save invalid nouns to pickled list
        try:
            with open('invalid_nouns.txt', 'wb') as fp:
                pickle.dump(self.invalid_nouns, fp)
        except:
            pass
    
    def load_invalid_nouns(self):
        ## load previously found invalid nouns from pickled list
        try:
            with open ('invalid_nouns.txt', 'rb') as fp:
                n = pickle.load(fp)
                self.invalid_nouns.extend(n)
        except:
            pass
    
    def init_word2vec(self):
        #model = Word2Vec.load('tutorial.model')
        f = open(self.tutorials_text, 'r')
        tutorials = f.read()
        sentences = word_tokenize(tutorials)
        w2v = Word2Vec([sentences])
        return w2v
        
    def init_tokenizer(self):
        #try: 
          #  self.load_tokenizer()
        #except EOFError:
        self.tokenizer = Tokenizer(num_words=self.vocab_size)
        
    def get_data(self, state, ):
        ## if we have generated actions before for state, load them, otherwise generate actions
        if (state in list(self.state_data['State'])):
            state_vector = list(self.state_data[self.state_data['State'] == state]['StateVector'])[0][0]
            try:
                actionsVectors = []
                actions = []
                probs = []
                action_dict = list(self.state_data[self.state_data['State'] == state]['ActionData'])[0]
                for act, data in action_dict.items():
                    actions.append(act)
                    probs.append(data[0])
                    actionsVectors.append(data[1])
                probs = np.array(probs)
            except:
                actionsVectors = []
                actions = []
                probs = []
                action_dict = list(self.state_data[self.state_data['State'] == state]['ActionData'])[0][0]
                for act, data in action_dict.items():
                    actions.append(act)
                    probs.append(data[0])
                    actionsVectors.append(data[1])
                probs = np.array(probs)
        else: 
    
            state_vector = self.vectorize_text(state,self.tokenizer)
            ## get nouns from state
            nouns = self.get_nouns(state)
            # build action space and probabilities 
            current_action_space = self.generate_action_tuples(nouns)
            action_space = set()
            action_space, probs = self.add_to_action_space(action_space, current_action_space)
            actions = []
            for a in action_space:
                actions.append(a)
            probs = np.array(probs)
            actionsVectors = []
            for a in actions:
                actionsVectors.append(self.vectorize_text(a,self.tokenizer))
            ## create action dictionary
            action_dict = dict()
            for idx, act in enumerate(actions):
                action_dict[act] = (probs[idx], actionsVectors[idx])
            ## store state data 
            row = len(self.state_data)
            self.state_data.loc[row, 'State'] = state
            self.state_data.loc[row, 'StateVector'] = [state_vector]
            self.state_data.loc[row, 'ActionData'] = [action_dict]
        return probs, actions, state_vector, actionsVectors, action_dict
    
    def perform_selected_action(self, action):
        self.perform_action(action)
        response,current_score,moves = self.readLine()
        response = self.preprocess(response)
        return response, current_score, moves

    def run_game(self, agent, num_games, num_rounds, batch_size):
        ## set global batch size
        self.batch_size = batch_size
        
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
                    if len(state) > self.state_limit:
                        print('encountered line read bug')
                        state, old_surroundings, old_inventory, _, _, = self.get_state()
                        
                    ## get data for current state
                    probs, actions, state_vector, actionsVectors, action_dict = self.get_data(state)
                    
                    ## decide which type of action to perform
                    if (agent.act_random()): ## choose random action
                        print('random choice:')
                        probs_norm = probs/(probs.sum())
                        action = self.select_one(actions, probs_norm)       
                    else: ## choose predicted max Q value action
                        print('predicted choice:')
                        action = agent.predict_actions(state_vector, action_dict)

                    print('-- ' + action + ' --')
    
                    ## perform selected action
                    response, current_score, moves = self.perform_selected_action(action)
                    
                    ## check for invalid nouns
                    invalid_noun = self.detect_invalid_nouns(response)

                    ## vectorize selected action
                    action_vector = self.vectorize_text(action,self.tokenizer)
    
                    ## check new state after performing action
                    new_state, surroundings, inventory, current_score, moves = self.get_state()
                    new_state = self.preprocess(new_state)
                    new_state_vector = self.vectorize_text(new_state, self.tokenizer)
                    
                    ## get reward
                    round_score = current_score - self.game_score
                    self.game_score = current_score
                    reward, reward_msg = self.calculate_reward(inventory, old_inventory, i, state, new_state, round_score)
    
                    ## remember round data
                    agent.remember(state_vector, action_vector, reward, new_state_vector, False)
                    
                    ## update story dataframe
                    self.score += reward
                    total_round_number = i + game_number*num_rounds
                    self.story.loc[total_round_number] = [state, old_inventory, action, response, reward, 
                                  reward_msg, self.score, str(i), total_round_number]
                    
                    ## remember already tried actions that don't change current game state
                    invalid_action = ''
                    if (reward==-1):
                        invalid_action = action
                    
                    ## check if we have an invalid noun or action and remove them from the action dictionary
                    if (invalid_noun or invalid_action):
                        if invalid_noun:
                            for act in list(action_dict.keys()):
                                    if invalid_noun in act:
                                        del action_dict[act]
                        if invalid_action and invalid_action in action_dict:
                            del action_dict[invalid_action]
                        ## update state data 
                        self.state_data.loc[self.state_data['State'] == state, 'ActionData'] = [action_dict]
                        
                    ## if enough experiences in batch, replay 
                    if (i+1)%self.batch_size == 0 and i>0:  
                        print('Training on mini batch')
                        self.agent.replay(self.batch_size)
                        sleep(self.sleep_time)
                        
                    ## update progress bar
                    pbar.update(i + (game_number)*num_rounds) 
                    
                self.end_game_scores.loc[game_number] = [game_number, self.score]
                self.restart_game()
            except Exception as e:
                print('exception')
                traceback.print_tb(e.__traceback__)
                self.restart_game()
                #print(e.with_traceback())
            pbar.finish()
            self.stories.append(self.story)
        self.state_data.to_pickle('state_data.pickle')
        self.kill_game()
        return True

