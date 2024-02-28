#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
from Environment import StochasticWindyGridworld
from Helper import argmax

class QValueIterationAgent:
    ''' Class to store the Q-value iteration solution, perform updates, and select the greedy action '''

    def __init__(self, n_states, n_actions, gamma, threshold=0.01):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.Q_sa = np.zeros((n_states,n_actions))
        
    def select_action(self,s):
        ''' Returns the greedy best action in state s ''' 
        # TO DO: Add own code
        a = argmax(self.Q_sa[s, :])
        return a
        
    def update(self,s,a,p_sas,r_sas):
        ''' Function updates Q(s,a) using p_sas and r_sas 
        and computes the absolute error in this sweep '''
        # TO DO: Add own code
        temp = self.Q_sa[s, a]
        self.Q_sa[s, a] = np.sum(p_sas * (r_sas + self.gamma * np.max(self.Q_sa, axis=1))) 
        error = np.abs(temp - self.Q_sa[s, a])   
        return error
    
    
def Q_value_iteration(env, gamma=1.0, threshold=0.001):
    ''' Runs Q-value iteration. Returns a converged QValueIterationAgent object '''
    
    QIagent = QValueIterationAgent(env.n_states, env.n_actions, gamma)
 
     # TO DO: IMPLEMENT Q-VALUE ITERATION HERE
    i = 0

    while True:
        i += 1
        max_error = 0  # Reset max_error for each iteration
        
        for s in range(QIagent.n_states):  
            for a in range(QIagent.n_actions):  
                p_sas, r_sas = env.model(s, a)
                error = QIagent.update(s, a, p_sas, r_sas)
                
                # Update max_error if the current error is larger
                max_error = max(max_error, error)

        # Plot current Q-value estimates & print max error
        env.render(Q_sa=QIagent.Q_sa,plot_optimal_policy=True,step_pause=0.2)
        print("Q-value iteration, iteration {}, max error {}".format(i,max_error))

        # Check for convergence
        if(max_error < threshold):
            break

    return QIagent


def experiment():
    gamma = 1.0
    threshold = 0.001
    env = StochasticWindyGridworld(initialize_model=True)
    env.render()
    QIagent = Q_value_iteration(env,gamma,threshold)
    print(max(QIagent.Q_sa[3, :]))
    # view optimal policy
    done = False
    s = env.reset()

    cum_reward = 0
    time_steps = 0

    while not done:
        a = QIagent.select_action(s)
        s_next, r, done = env.step(a)
        env.render(Q_sa=QIagent.Q_sa,plot_optimal_policy=True,step_pause=0.5)
        s = s_next

        cum_reward += r
        time_steps += 1

        mean_reward_per_timestep = cum_reward / time_steps

        # TO DO: Compute mean reward per timestep under the optimal policy
        print("Mean reward per timestep under optimal policy: {}".format(mean_reward_per_timestep))
    
if __name__ == '__main__':
    experiment()
