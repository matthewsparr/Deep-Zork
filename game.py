
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
        self.vocab_size = 800
        
        self.sleep_time = 0.01
        self.random_action_weight = 6
        self.random_action_basic_prob = 0.5
        self.random_action_low_prob = 0.2
        
        self.game_score_weight = 30
        self.negative_per_turn_reward = 1
        self.inventory_reward_value = 30
        self.new_area_reward_value = 1
                
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
        self.story = pd.DataFrame(columns=['Surroundings', 'Inventory', 'Action', 'Response', 'Score', 'Moves'])
        self.stories = []
        
    def enqueue_output(self, out, queue):
        for line in iter(out.readline, b''):
            queue.put(line)
        out.close()

    def start_game(self, game_file):
        self.load_actions_probs_dict()
        self.load_invalid_nouns()
        self.load_unique_surroundings()
        self.init_word2vec()
        self.init_tokenizer()
        
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
        self.kill_game()
        
    def kill_game(self):
        self.p.terminate()
        self.p.kill()

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
        
    def preprocess(self, text):
        # fix bad newlines (replace with spaces), unify quotes
        text = text.strip()
        text = text.replace('\\n', ' ').replace('‘', '\'').replace('’', '\'').replace('”', '"').replace('“', '"')
        # convert to lowercase
        text = text.lower()
        # remove all characters except alphanum, spaces and - ' "
        text = self.compiled_expression.sub('', text)
        # split numbers into digits to avoid infinite vocabulary size if random numbers are present:
        #text = re.sub('[0-9]', ' \g<0> ', text)
        # expand unambiguous 'm, 't, 're, ... expressions
        text = text.replace('\'m ', ' am ').replace('\'re ', ' are ').replace('won\'t', 'will not').replace('n\'t', ' not').replace('\'ll ', ' will ').replace('\'ve ', ' have ').replace('\'s', ' \'s')
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
    
    def calculate_reward(self, story, moves_count, new_narrative):
        reward = 0

        ## add reward from score in game
        if(moves_count != 0):
            new_score = int(story['Score'][moves_count]) - int(story['Score'][moves_count-1])
            reward = reward + new_score*self.game_score_weight
            if (new_score > 0):
                print('Scored ' + str(new_score) + ' points in game.')

        ## add small negative reward for each move
        reward = reward - self.negative_per_turn_reward

        ## add reward for picking up / using items
        if(moves_count != 0):
            pre_inventory = story['Inventory'][moves_count-1]
            inventory = story['Inventory'][moves_count]
            
            if (pre_inventory != inventory) and ('chirping' not in inventory):  ## inventory changed, ignoring chirping bird line
                reward = reward + self.inventory_reward_value
                print('inventory changed')


        ## add reward for discovering new areas
        if new_narrative not in self.unique_narratives:  ## new location
            reward = reward + self.new_area_reward_value
            self.unique_narratives.add(new_narrative)
            print('discovered new area')
        print('Rewarded: ' + str(reward) + ' points.')
        return reward

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
            try:
                for i in range(0, num_rounds):
                    ## Check surroundings, check inventory, choose action, check action response
                    narrative,score,moves = self.readLine(self.q)
                    self.check_inventory(self.p)
                    inventory,s,m = self.readLine(self.q)
                    ## if first round need to re-check surroundings to get initial environment
                    if (i==0):
                        self.look_surroundings(self.p)
                        narrative,score,moves = self.readLine(self.q)
                        narrative = self.preprocess(narrative)
                    else: 
                        narrative = self.preprocess(narrative)
                        
                    ## check if data stored in history about this environment
                    if (narrative in self.actions_probs_dict):
                        actions, probs, narrativeVector, actionsVectors = self.actions_probs_dict.get(narrative)
                    else:
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
                    
                    print(narrative)
                    ## decide which type of action to perform
                    if (agent.act_random() or i < 10): ## choose random action
                        print('random choice:')
                        action = self.select_one(actions, probs)
                    else: ## choose predicted max Q value action
                        print('predicted choice:')
                        best_action, max_q = agent.predict_actions(narrativeVector, actionsVectors)
                        action = actions[best_action]
                    print('-- ' + action + ' --')
    
                    ## perform selected action
                    self.perform_action(action, self.p)
    
                    ## grab response from action
                    response,score,moves = self.readLine(self.q)
                    response = self.preprocess(response)
                    
                    ## check for invalid nouns
                    self.detect_invalid_nouns(response)
                    
                    ## update story dataframe
                    self.story.loc[(i + game_number*num_rounds)] = [narrative, inventory, action, response, str(s), str(i+1)]
                    
                    ## update unique narratives set
                    self.unique_narratives.add(narrative)
    
                    ## vectorize selected action
                    actionVector = self.vectorize_text(action,self.tokenizer)
    
                    ## check new surroundings after performing action
                    self.look_surroundings(self.p)
                    new_narrative,s,m = self.readLine(self.q)
                    new_narrative = self.preprocess(new_narrative)
                    new_narrativeVector = self.vectorize_text(new_narrative, self.tokenizer)
    
                    ## get reward
                    reward = self.calculate_reward(self.story, i, new_narrative)
    
                    ## remember round data
                    agent.remember(narrativeVector, actionVector, reward, new_narrativeVector, False)
    
                    ## check new surroundings
                    self.look_surroundings(self.p)
    
                    ## if enough experiences in batch, replay 
                    if i%self.batch_size == 0 and i>0:  
                        print('Training on mini batch')
                        self.agent.replay(self.batch_size)
                        
                    ## update progress bar
                    pbar.update(i + (game_number)*num_rounds) 
                    
                self.end_game()
            except Exception as e:
                print('exception')
                traceback.print_tb(e.__traceback__)
                self.kill_game()
                #print(e.with_traceback())
            pbar.finish()
            try:
                self.kill_game()
            except:
                print('finished - killing game')
            self.stories.append(self.story)
        return self.story

