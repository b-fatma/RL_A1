#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
from Environment import StochasticWindyGridworld
from Agent import BaseAgent

class NstepQLearningAgent(BaseAgent):
        
    def update(self, states, actions, rewards, done, n):
        ''' states is a list of states observed in the episode, of length T_ep + 1 (last state is appended)
        actions is a list of actions observed in the episode, of length T_ep
        rewards is a list of rewards observed in the episode, of length T_ep
        done indicates whether the final s in states is was a terminal state '''
        # TO DO: Add own code
        T_ep = len(states) - 1  # Length of the episode
        for t in range(T_ep):
            G = 0  # Reset cumulative return for each time step
            m = min(n, T_ep-t)
            for i in range(m):
                G += (self.gamma ** i) * rewards[t + i]  # Use rewards[t + i]
            if t + n < T_ep or not done:
                G += (self.gamma ** m) * np.max(self.Q_sa[states[t+m], :])
            temp = self.Q_sa[states[t], actions[t]]
            self.Q_sa[states[t], actions[t]] += self.learning_rate * (G - temp) 
        pass

def n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy='egreedy', epsilon=None, temp=None, plot=True, n=5, eval_interval=500):
    ''' runs a single repetition of an MC rl agent
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    eval_env = StochasticWindyGridworld(initialize_model=False)
    pi = NstepQLearningAgent(env.n_states, env.n_actions, learning_rate, gamma)
    eval_timesteps = []
    eval_returns = []

    # TO DO: Write your n-step Q-learning algorithm here!
    timestep = 0
    while timestep < n_timesteps:
        s = env.reset()
        states = [s]
        rewards = []
        actions = []
        for t in range(max_episode_length):
            a = pi.select_action(s=s, policy=policy, epsilon=epsilon, temp=temp)
            s_next, r, done = env.step(a)
            states.append(s_next)
            rewards.append(r)
            actions.append(a)
            s = s_next
            timestep += 1
            if plot:
                env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=0.1) # Plot the Q-value estimates during n-step Q-learning execution
            if timestep % eval_interval == 0:
                eval_timesteps.append(timestep)
                eval_returns.append(pi.evaluate(eval_env))
            if done:
                env.reset()
                break

        pi.update(states, actions, rewards, done, n)
         
    return np.array(eval_returns), np.array(eval_timesteps) 

def test():
    n_timesteps = 500001
    max_episode_length = 1000
    gamma = 0.1
    learning_rate = 0.1
    n = 5
    
    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.2
    temp = 1.0
    
    # Plotting parameters
    plot = False
    print(n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy, epsilon, temp, plot, n=n))
    
    
if __name__ == '__main__':
    test()
