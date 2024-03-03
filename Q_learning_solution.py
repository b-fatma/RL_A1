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

class QLearningAgent(BaseAgent):
        
    def update(self,s,a,r,s_next,done):
        # TO DO: Add own code
        target = r + self.gamma * np.max(self.Q_sa[s_next, :]) * (1 - done)
        self.Q_sa[s, a] += self.learning_rate * (target - self.Q_sa[s, a])
        pass

def q_learning(n_timesteps, learning_rate, gamma, policy='egreedy', epsilon=None, temp=None, plot=True, eval_interval=500):
    ''' runs a single repetition of q_learning
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    eval_env = StochasticWindyGridworld(initialize_model=False)
    agent = QLearningAgent(env.n_states, env.n_actions, learning_rate, gamma)
    eval_timesteps = []
    eval_returns = []
    
    # TO DO: Write your Q-learning algorithm here!
    state = env.reset()
    for t in range(n_timesteps):
        
        if policy=='greedy':
            a = agent.select_action(state, 'greedy')
        elif policy == 'egreedy':
            a = agent.select_action(state, 'egreedy', epsilon=epsilon)
        elif policy == 'softmax':
            a = agent.select_action(state, 'softmax', temp=temp)

        next_state, reward, done = env.step(a)
            
        # Update Q-values using the Q-learning update rule
        agent.update(state, a, reward, next_state, done)

        if done:
                env.reset()
        else:
            state = next_state

        if plot:
            env.render(Q_sa=agent.Q_sa, plot_optimal_policy=True, step_pause=0.001) 
                
        if t % eval_interval == 0:
            eval_return = agent.evaluate(eval_env)
            eval_timesteps.append(t)
            eval_returns.append(eval_return)


    return np.array(eval_returns), np.array(eval_timesteps)


def test():
    
    n_timesteps = 5000
    eval_interval = 1000
    gamma = 1.0
    learning_rate = 0.2

    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = True

    eval_returns, eval_timesteps = q_learning(n_timesteps, learning_rate, gamma, policy, epsilon=epsilon, temp=temp, plot=plot, eval_interval=eval_interval)
    print(eval_returns,eval_timesteps)

if __name__ == '__main__':
    test()
