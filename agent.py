
# coding: utf-8

# In[2]:
from collections import deque
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Dense, Dot
import numpy as np
from keras.preprocessing.sequence import pad_sequences

# Deep Q-learning Agent
class DQNAgent:
    def __init__(self):
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.998
        self.learning_rate = 0.001
        self.vocab_size = 500
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
        dot_state_action = Dot(axes=-1, normalize=True, name="dot_state_action")([input_dot_state, input_dot_action])

        model_dot_state_action = Model(inputs=[input_dot_state, input_dot_action], outputs=dot_state_action,
                                           name="dot_state_action")
        self.model_dot_state_action = model_dot_state_action

        model_state = Model(inputs=input_state, outputs=dense_state, name="state")
        model_action = Model(inputs=input_action, outputs=dense_action, name="action")
        self.model_state = model_state
        self.model_action = model_action

        model = Model(inputs=[model_state.input, model_action.input], 
                      outputs=model_dot_state_action([model_state.output, model_action.output]))

        model.compile(optimizer='Adam', loss='mse')
        return model
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    def act_random(self):
        ## decides to perform either a random action or an action with the highest Q value
        if np.random.rand() <= self.epsilon:
            return True
        else:
            return False
    def predict_actions(self, state_vector, action_dict):
        q_max = -np.math.inf
        best_action = 0
        
        for action, data in action_dict.items():

            action_vector = data[1]           
            state_dense = self.model_state.predict([state_vector])[0]
            action_dense = self.model_action.predict([action_vector])[0]

            q = self.model_dot_state_action.predict([state_dense.reshape((1, len(state_dense))), 
                                                     action_dense.reshape((1, len(action_dense)))])[0][0]

            if q > q_max:
                q_max = q
                best_action = action
                print(action, q)

        return best_action
        
    def replay(self, batch_size):
        states = [None]*batch_size
        actions = [None]*batch_size
        targets = np.zeros((batch_size, 1))
        for i in range(0, batch_size):
            state, action, reward, next_state, done = self.memory.popleft()
            target = reward
            if not done:
                state_dense = self.model_state.predict([state])[0]
                action_dense = self.model_action.predict([action])[0]
                q = self.model_dot_state_action.predict([state_dense.reshape((1, len(state_dense))), action_dense.reshape((1, len(action_dense)))])[0][0]
                target = reward + self.gamma*q
                states[i] = state[0]
                actions[i] = action[0]
                targets[i] = target
                
            #target_f = self.model.predict([state, action])
            #target_f[0][action] = target
        states = pad_sequences(states)
        actions = pad_sequences(actions)
        self.model.fit(x=[states, actions], y=targets, batch_size=batch_size, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

