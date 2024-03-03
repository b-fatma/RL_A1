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

class MonteCarloAgent(BaseAgent):
        
    def update(self, states, actions, rewards):
        ''' states is a list of states observed in the episode, of length T_ep + 1 (last state is appended)
        actions is a list of actions observed in the episode, of length T_ep
        rewards is a list of rewards observed in the episode, of length T_ep
        done indicates whether the final s in states is was a terminal state '''
        # TO DO: Add own code
        T_ep = len(states) - 1  # Length of the episode

        G = 0  # Initialize return

        # Update Q-values for each state-action pair in reverse order
        for t in range(T_ep - 1, -1, -1):
            G = self.gamma * G + rewards[t]
            Q_s_a = self.Q_sa[states[t]][actions[t]]
            delta = G - Q_s_a
            self.Q_sa[states[t]][actions[t]] += self.learning_rate * delta

def monte_carlo(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy='egreedy', epsilon=None, temp=None, plot=True, eval_interval=500):
    ''' runs a single repetition of an MC rl agent
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    eval_env = StochasticWindyGridworld(initialize_model=False)
    pi = MonteCarloAgent(env.n_states, env.n_actions, learning_rate, gamma)
    eval_timesteps = []
    eval_returns = []

    # TO DO: Write your Monte Carlo RL algorithm here!
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

        pi.update(states, actions, rewards)          
    return np.array(eval_returns), np.array(eval_timesteps) 
    
def test():
    n_timesteps = 1000
    max_episode_length = 100
    gamma = 1.0
    learning_rate = 0.1

    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = True

    monte_carlo(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy, epsilon, temp, plot)
    
            
if __name__ == '__main__':
    test()
