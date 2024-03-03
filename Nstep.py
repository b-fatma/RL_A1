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
    for t in range(1, n_timesteps + 1):
        state = env.reset()
        episode_states = [state]
        episode_actions = []
        episode_rewards = []
        done = False

        # Collect an episode
        for _ in range(max_episode_length):
            if policy=='greedy':
                action = pi.select_action(state, 'greedy')
            elif policy == 'egreedy':
                action = pi.select_action(state, 'egreedy', epsilon=epsilon)
            elif policy == 'softmax':
                action = pi.select_action(state, 'softmax', temp=temp)
            next_state, reward, done = env.step(action)
            episode_states.append(next_state)
            episode_actions.append(action)
            episode_rewards.append(reward)

            state = next_state

            if plot:
                env.render(Q_sa=pi.Q_sa, plot_optimal_policy=True, step_pause=0.1)

            if done:
                break

        # Update Q-values using n-step Q-learning
        pi.update(episode_states, episode_actions, episode_rewards, done, n)

        # Evaluate the agent at regular intervals
        if t % eval_interval == 0:
            eval_return = pi.evaluate(eval_env)
            eval_returns.append(eval_return)
            eval_timesteps.append(t)

        

    # if plot:
    #    env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=0.1) # Plot the Q-value estimates during n-step Q-learning execution
        
    return np.array(eval_returns), np.array(eval_timesteps) 

def test():
    n_timesteps = 10000
    max_episode_length = 100
    gamma = 1.0
    learning_rate = 0.1
    n = 5
    
    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.2
    temp = 1.0
    
    # Plotting parameters
    plot = True
    n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy, epsilon, temp, plot, n=n)
    
    
if __name__ == '__main__':
    test()
