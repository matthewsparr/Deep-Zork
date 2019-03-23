# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 22:21:04 2019

@author: sparr
"""

actions_to_remove = []

                            for invalid in already_tried_actions:
                                if invalid in actions:
                                    actions_to_remove.append(invalid)
                            revised_actions = actions.copy()

                            revised_probs = probs.copy()
                            if len(actions_to_remove)>0:
                                for x in actions_to_remove:
                                    idx = revised_actions.index(x)
                                    revised_actions.remove(x)
                                    revised_probs = np.delete(revised_probs, idx)
                            revised_probs /= revised_probs.sum()


            'Actions', 'Probs', 'ActionsVectors'
                        probs /= probs.sum()

if(len(action_dict)<1):
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
                            
                            
                         20 minutes per 10/256 random game
                         15 minutes per 1/256 predicted game