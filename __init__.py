from subprocess import Popen, PIPE, STDOUT
from random import randint
import binascii
import os
from queue import Queue, Empty
from threading  import Thread
from time import sleep
import random
import nltk
nltk.download('averaged_perceptron_tagger')
from itertools import permutations, combinations
import spacy
from spacy.matcher import Matcher
from spacy.attrs import POS
nlp = spacy.load('en_core_web_sm')
import pandas as pd
nltk.download('punkt')
import numpy as np
import gensim
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize, word_tokenize
import timeit
from keras.models import Model
from keras.preprocessing.text import one_hot, Tokenizer
from keras.preprocessing.sequence import pad_sequences
import re
from agent import DQNAgent as DQNAgent
import importlib
from collections import deque
from numpy.random import choice
import math
import pickle
from progressbar import ProgressBar
from agent import DQNAgent
from game_commands import commands