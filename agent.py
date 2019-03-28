
# coding: utf-8

# In[2]:
from collections import deque
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Dense, Dot
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import random
import tensorflow as tf
import pandas as pd
# Deep Q-learning Agent
class DQNAgent:
    def __init__(self):
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9998
        self.learning_rate = 0.1
        self.discount_factor = 0.75
        self.vocab_size = 1200
        self.state_q_values = dict()
        self.model_histories = list()
        self.model = self._build_model()
    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        embed_dim = 16
        lstm_dim = 32 
        dense_dim = 8

        input_state = Input(batch_shape=(None, None), name="input_state")
        input_action = Input(batch_shape=(None, None), name="input_action")

        embedding_shared = Embedding(self.vocab_size + 1, embed_dim, input_length=None, mask_zero=True,
                            trainable=True, name="embedding_shared")
        embedding_state = embedding_shared(input_state)
        embedding_action = embedding_shared(input_action)

        lstm_shared = LSTM(lstm_dim, name="lstm_shared")
        lstm_state = lstm_shared(embedding_state)
        lstm_action = lstm_shared(embedding_action)

        dense_state = Dense(dense_dim, activation='tanh', name="dense_state")(lstm_state)
        dense_action = Dense(dense_dim, activation='tanh', name="dense_action")(lstm_action)

        input_dot_state = Input(shape=(dense_dim,))
        input_dot_action = Input(shape=(dense_dim,))
        dot_state_action = Dot(axes=-1, normalize=False, name="dot_state_action")([input_dot_state, input_dot_action])

        model_dot_state_action = Model(inputs=[input_dot_state, input_dot_action], outputs=dot_state_action,
                                           name="dot_state_action")
        self.model_dot_state_action = model_dot_state_action

        model_state = Model(inputs=input_state, outputs=dense_state, name="state")
        model_action = Model(inputs=input_action, outputs=dense_action, name="action")
        self.model_state = model_state
        self.model_action = model_action

        model = Model(inputs=[model_state.input, model_action.input], 
                      outputs=model_dot_state_action([model_state.output, model_action.output]))

        model.compile(optimizer='RMSProp', loss='mse')
        return model
    def remember(self, state, state_text, action, reward, next_state, next_state_text, action_dict, done):
        self.memory.append((state, action, reward, next_state, next_state_text, action_dict, done))
    def act_random(self):
        ## decides to perform either a random action or an action with the highest Q value
        if np.random.rand() <= self.epsilon:    
            return True
        else:
            return False
    def predict_actions(self, state_text, state, action_dict):
        state_dense = self.model_state.predict([state])[0]
        state_input = state_dense.reshape((1, len(state_dense)))
        best_action, q_max = self.compute_max_q(state_text, state_input, action_dict)
        return best_action
    
    def compute_max_q(self, state_text, state_input, action_dict):
        if state_text in self.state_q_values:
            q_target = self.state_q_values[state_text]
        else:
            q_target = 0
            self.state_q_values[state_text] = q_target
        i = 0
        q_max = -np.math.inf
        action_items = list(action_dict.items())
        index_list = list(range(0,len(action_items)))
        random.shuffle(index_list)
        while (q_max < q_target and i < len(action_items)):
            idx = index_list[i]
            action, data = action_items[idx]
            action_vector = data[1]           
            action_dense = self.model_action.predict([action_vector])[0]
            action_input = action_dense.reshape((1, len(action_dense)))
            with tf.device('/gpu:0'):
                q = self.model_dot_state_action.predict([state_input, action_input], batch_size=1)[0][0]
            if q > q_max:
                q_max = q
                best_action = action
                print(action, q)
            i += 1
        self.state_q_values[state_text] = q_max
        return best_action, q_max
        
    def replay(self, batch_size):
        states = [None]*batch_size
        actions = [None]*batch_size
        targets = np.zeros((batch_size, 1))
        next_state_dict = pd.DataFrame(columns=['next_state', 'next_state_input', 'future_q'])
        for i in range(0, batch_size):
            state, action, reward, next_state, next_state_text, action_dict, done = self.memory.popleft()
            target = reward
            if not done:
                ## calculate current q
                state_dense = self.model_state.predict([state])[0]
                state_input = state_dense.reshape((1, len(state_dense)))
                action_dense = self.model_action.predict([action])[0]
                action_input = action_dense.reshape((1, len(action_dense)))

                with tf.device('/gpu:0'):
                    q = self.model_dot_state_action.predict([state_input, action_input])[0][0]
                ## calculate maximum future q
                try:
                    next_state_input = next_state_dict[next_state_dict['next_state'] == next_state]['next_state_input'] 
                    future_q = next_state_dict[next_state_dict['next_state'] == next_state]['future_q']
                except:
                    with tf.device('/gpu:0'):
                        next_state_dense = self.model_state.predict([next_state])[0]
                        next_state_input = next_state_dense.reshape((1, len(state_dense)))
                        best_action, future_q = self.compute_max_q(next_state_text, next_state_input, action_dict)
                        row = len(next_state_dict)
                        next_state_dict.loc[row, 'next_state'] = next_state
                        next_state_dict.loc[row, 'next_state_input'] = next_state_input
                        next_state_dict.loc[row, 'future_q'] = future_q

                ## calculate target
                target = q + self.learning_rate*(reward + self.discount_factor*future_q - q)
                
                ## store state, action, target
                states[i] = state[0]
                actions[i] = action[0]
                targets[i] = target
                
        states = pad_sequences(states)
        actions = pad_sequences(actions)
        history = self.model.fit(x=[states, actions], y=targets, 
                                 batch_size=batch_size, epochs=1, verbose=1)
        self.model_histories.append(history)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

