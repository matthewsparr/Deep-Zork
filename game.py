
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
from gensim.models import Word2Vec
from progressbar import ProgressBar
import numpy as np
from keras.preprocessing.text import Tokenizer
import pandas as pd
from time import sleep
import traceback


class text_game:
    
    def __init__(self):
        from agent import DQNAgent
        from game_commands import commands
        import pandas as pd
        import spacy
        import re
        
        self.emulator_file = 'dfrotz.exe'
        
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
        self.vocab_size = 1000
        self.narrative_limit = 500
        
        self.sleep_time = 0.25
        
        self.random_action_weight = 6
        self.random_action_basic_prob = 0.5
        self.random_action_low_prob = 0.2
        
        self.score = 0
        self.game_score = 0
        self.game_score_weight = 100
        self.negative_per_turn_reward = 1
        self.inventory_reward_value = 100
        self.new_area_reward_value = 10
                
        cmds = commands()
        self.basic_actions = cmds.basic_actions
        self.directions = cmds.directions
        self.command1_actions = cmds.command1_actions
        self.command2_actions = cmds.command2_actions
        self.action_space = cmds.action_space
        self.filtered_tokens = cmds.filtered_tokens
        self.invalid_nouns = [] 
        
        self.unique_narratives = set()
        self.unique_surroundings = dict()
        self.actions_probs_dict = dict()
        self.invalid_actions_dict = dict()
        self.story = pd.DataFrame(columns=['Surroundings', 'Inventory', 'Action', 'Response', 'Reward', 'Reward_Type', 'Score', 'Moves', 'Total_Moves'])
        self.stories = []
        
        self.load_actions_probs_dict()
        self.load_invalid_nouns()
        self.load_invalid_actions_dict()
        self.load_unique_surroundings()
        self.init_word2vec()
        self.init_tokenizer()
        
    def enqueue_output(self, out, queue):
        for line in iter(out.readline, b''):
            queue.put(line)
        out.close()

    def start_game(self, game_file):
        self.score = 0
        self.unique_narratives = set()
        self.game_score = 0
        self.p = Popen([self.emulator_file, game_file], stdin=PIPE, stdout=PIPE, stderr=PIPE, shell=True)
        self.q = Queue()
        self.t = Thread(target=self.enqueue_output, args=(self.p.stdout, self.q))
        self.t.daemon = True # thread dies with the program
        self.t.start()
        print('___starting new game')
        sleep(self.sleep_time*10)
        return()
    
    def end_game(self):
        self.save_invalid_nouns()
        self.save_unique_surroundings()
        self.save_actions_probs_dict()
        self.save_invalid_actions_dict()
        self.kill_game()
        
    def kill_game(self):
        self.save_invalid_nouns()
        self.save_unique_surroundings()
        self.save_actions_probs_dict()
        self.p.terminate()
        self.p.kill()
        
    def restart_game(self):
        self.save_invalid_nouns()
        self.save_unique_surroundings()
        self.save_actions_probs_dict()
        self.save_invalid_actions_dict()
        self.perform_action('restart', self.p)
        self.score = 0
        self.unique_narratives = set()
        self.game_score = 0
        
    # read line without blocking
    def readLine(self, q):
        cont = True
        narrative = ""
        while cont:
            try:  line = self.q.get_nowait() # or q.get(timeout=.1)
            except Empty:
                cont = False
            else: 
                narrative = narrative + line.decode("utf-8").replace('\n', " ").replace('\r', " ")
        if ('840726' in narrative): ## Opening narrative
            narrative = narrative[narrative.index('840726') + len('840726'):]
        try:
            score, moves = self.grab_score_moves(narrative)
            narrative = narrative[narrative.index('Moves: ')+len('Moves:')+5:-1].strip()
        except:  ## not valid move
            pass
        sleep(self.sleep_time)
        return(narrative, score, moves)

    def grab_score_moves(self, narrative):
        try:
            score = int(narrative[narrative.index('Score: ') + len('Score: '):][0:3].strip())
            moves = int(narrative[narrative.index('Moves: ') + len('Moves: '):][0:3].strip())
        except:  ## not valid move
            score = 0
            moves = 0
        return(score, moves)

    def look_surroundings(self, p):
        self.perform_action('look', p)

    def check_inventory(self, p):
        self.perform_action('inventory', p)

    def get_nouns(self, narrative):
        matcher = Matcher(self.nlp.vocab)
        matcher.add('Noun phrase', None, [{POS: 'NOUN'}])
        doc = self.nlp(narrative)
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
                            similarities.append(self.word_2_vec.similarity(word_tokenize(action_to_add)[0], i))
                        except:
                            similarities.append(self.random_action_low_prob)

        return possible_actions
    
    def select_one(self, action_space, similarities):
        random_index = choice(len(action_space), p=similarities)
        return action_space[random_index]
    def add_to_action_space(self, action_space, actions):
        ## 
        
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
        
    def perform_action(self, command, p):
        p.stdin.write(bytes(command+ "\n", 'ascii'))
        p.stdin.flush()
        sleep(self.sleep_time)## wait for action to register
        
    def preprocess(self, text, check_for_flavor_text=False):
        
        # fix bad newlines (replace with spaces), unify quotes
        text = text.strip()
        text = text.replace('\\n', ' ').replace('‘', '\'').replace('’', '\'').replace('”', '"').replace('“', '"')
        # convert to lowercase
        text = text.lower()
        # remove all characters except alphanum, spaces and - ' "
        text = self.compiled_expression.sub('', text)
        # split numbers into digits to avoid infinite vocabulary size if random numbers are present:
        #text = re.sub('[0-9]', ' \g<0> ', text)
        
        ## remove problematic flavor text
        if check_for_flavor_text:
            flavor_text = 'you hear in the distance the chirping of a song bird'
            if (flavor_text in text):
                text = text.replace(flavor_text, '')
        
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
    
    def calculate_reward(self, inventory, old_inventory, moves_count, new_narrative, round_score):
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
            if  inventory.strip().lower() not in old_inventory.strip().lower():  ## inventory changed, ignoring chirping bird line
                reward = reward + self.inventory_reward_value
                print('inventory changed')
                reward_msg += ' inventory score (' + old_inventory + " --- " + inventory + ')'

        ## add reward for discovering new areas
        if new_narrative.strip() not in self.unique_narratives:  ## new location
            reward = reward + self.new_area_reward_value
            self.unique_narratives.add(new_narrative)
            reward_msg += ' new area score ---' + new_narrative.strip()

        print('Rewarded: ' + str(reward) + ' points.')
        return reward, reward_msg

    def detect_invalid_nouns(self, action_response):
        ## detect and remove invalid nouns from future turns
        if('know the word' in action_response):
            startIndex = action_response.find('\"')
            endIndex = action_response.find('\"', startIndex + 1)
            word = action_response[startIndex+1:endIndex]
            print('Didn\'t know the word: ' + word)
            self.invalid_nouns.append(word)

    def save_actions_probs_dict(self):
        ## save invalid nouns to pickled list
        try:
            with open('actions_probs_dict.txt', 'wb') as fp:
                pickle.dump(self.actions_probs_dict, fp)
        except:
            pass
    
    def load_actions_probs_dict(self):
        ## load previously found invalid nouns from pickled list
        try:
            with open ('actions_probs_dict.txt', 'rb') as fp:
                n = pickle.load(fp)
                self.actions_probs_dict.extend(n)
        except:
            pass
        
    def save_invalid_actions_dict(self):
        ## save invalid nouns to pickled list
        try:
            with open('invalid_actions_dict.txt', 'wb') as fp:
                pickle.dump(self.invalid_actions_dict, fp)
        except:
            pass
    
    def load_invalid_actions_dict(self):
        ## load previously found invalid nouns from pickled list
        try:
            with open ('invalid_actions_dict.txt', 'rb') as fp:
                n = pickle.load(fp)
                self.invalid_actions_dict.extend(n)
        except:
            pass
        
            
    def save_unique_surroundings(self):
        ## save invalid nouns to pickled list
        try:
            with open('unique_surroundings.txt', 'wb') as fp:
                pickle.dump(self.unique_surroundings, fp)
        except:
            pass
    
    def load_unique_surroundings(self):
        ## load previously found invalid nouns from pickled list
        try:
            with open ('unique_surroundings.txt', 'rb') as fp:
                n = pickle.load(fp)
                self.unique_surroundings.extend(n)
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
        self.tokenizer = Tokenizer(num_words=self.vocab_size)

    def run_game(self, agent, num_games, num_rounds, batch_size):
        self.batch_size = batch_size
        pbar = ProgressBar(maxval=num_rounds*num_games)
        pbar.start()
        for game_number in range(0, num_games):
            self.start_game('zork1.z5')
            new_narrative = ''
            inventory = ''
            already_tried_actions = []
            try:
                for i in range(0, num_rounds):
                    
                    if (i==0):
                        ## Check surroundings, check inventory, choose action, check action response
                        narrative,score,moves = self.readLine(self.q)
                        narrative = self.preprocess(narrative)
                        self.check_inventory(self.p)
                        inventory,s,m = self.readLine(self.q)
                        self.unique_narratives.add(narrative)
                    else:
                        narrative = new_narrative
                        
                    if narrative in self.invalid_actions_dict:
                        already_tried_actions = self.invalid_actions_dict[narrative]
                    
                    if len(narrative) > self.narrative_limit:
                        print('encountered bug')
                        self.look_surroundings(self.p)
                        narrative,score,moves = self.readLine(self.q)
                        narrative = self.preprocess(narrative)
                        self.check_inventory(self.p)
                        inventory,s,m = self.readLine(self.q)
                    
                    ## check if data stored in history about this environment
                    if (narrative in self.actions_probs_dict):
                        actions, probs, narrativeVector, actionsVectors = self.actions_probs_dict.get(narrative)
                    else:
                        print(narrative)
                        nouns = self.get_nouns(narrative)
                        # build action space
                        current_action_space = self.generate_action_tuples(nouns)
                        action_space = set()
                        action_space, probs = self.add_to_action_space(action_space, current_action_space)
                        actions = []
                        for a in action_space:
                            actions.append(a)
                        probs = np.array(probs)
                        probs /= probs.sum()
                        narrativeVector = self.vectorize_text(narrative,self.tokenizer)
                        actionsVectors = []
                        for a in actions:
                            actionsVectors.append(self.vectorize_text(a,self.tokenizer))
                        self.actions_probs_dict[narrative] = actions,probs,narrativeVector,actionsVectors
                    
                    ## decide which type of action to perform
                    ## if chosen action has invalid noun, repick 
                    
                    
                    validNounFlag = True
                    nonRepeatedActionFlag = True
                    while(validNounFlag and nonRepeatedActionFlag):
                        if (agent.act_random()): ## choose random action
                            print('random choice:')
                            action = self.select_one(actions, probs)
                            
                        else: ## choose predicted max Q value action
                            print('predicted choice:')

                            action, max_q = agent.predict_actions(narrativeVector, actionsVectors, 
                                                                  actions, already_tried_actions)
                        
                        nonRepeatedActionFlag = False
                        if action in already_tried_actions:
                            nonRepeatedActionFlag = False
                            print(already_tried_actions)
                            break
                        validNounFlag = False
                        for noun in self.invalid_nouns:
                            if noun in action:
                                validNounFlag = True
                                break
                    print('-- ' + action + ' --')
    
                    ## perform selected action
                    self.perform_action(action, self.p)
    
                    ## grab response from action
                    response,current_score,moves = self.readLine(self.q)
                    response = self.preprocess(response)
                    
                    ## check for invalid nouns
                    self.detect_invalid_nouns(response)

                    ## vectorize selected action
                    actionVector = self.vectorize_text(action,self.tokenizer)
    
                    ## check new surroundings after performing action
                    self.look_surroundings(self.p)
                    new_narrative,s,m = self.readLine(self.q)
                    new_narrative = self.preprocess(new_narrative, check_for_flavor_text=True)
                    new_narrativeVector = self.vectorize_text(new_narrative, self.tokenizer)
                    
                    ## check inventory after performing action
                    self.check_inventory(self.p)
                    old_inventory = inventory
                    inventory,s,m = self.readLine(self.q)
                    inventory = self.preprocess(inventory, check_for_flavor_text=True)
    
                    ## get reward
                    round_score = current_score - self.game_score
                    reward, reward_msg = self.calculate_reward(inventory, old_inventory, i, new_narrative, round_score)
    
                    ## remember round data
                    agent.remember(narrativeVector, actionVector, reward, new_narrativeVector, False)
                    
                    ## update story dataframe
                    self.score += reward
                    total_round_number = i + game_number*num_rounds
                    self.story.loc[total_round_number] = [narrative, old_inventory, action, response, reward, reward_msg, self.score, str(i), total_round_number]
                    
                    
                    ## remember already tried actions that don't change current game state
                    if(new_narrative.strip() in narrative.strip()) and (inventory.strip() in old_inventory.strip()):
                        already_tried_actions.append(action)
                        self.invalid_actions_dict[narrative] = already_tried_actions
                        self.save_invalid_actions_dict()
                                            
                    ## if enough experiences in batch, replay 
                    if (i+1)%self.batch_size == 0 and i>0:  
                        print('Training on mini batch')
                        self.agent.replay(self.batch_size)
                        sleep(self.sleep_time)
                        
                    ## update progress bar
                    pbar.update(i + (game_number)*num_rounds) 
                    
                self.restart_game()
            except Exception as e:
                print('exception')
                traceback.print_tb(e.__traceback__)
                self.restart_game()
                #print(e.with_traceback())
            pbar.finish()
            self.stories.append(self.story)
        return True

