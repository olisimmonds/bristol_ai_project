# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 16:23:25 2023

@author: maxro
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import gymnasium as gym
import scipy.signal
import time
from bayes_opt import BayesianOptimization


"""
## Functions and class
"""


def discounted_cumulative_sums(x, discount):
    # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

def set_up_class(gamma, lam):
    class Buffer:
        # Buffer for storing trajectories
        # def __init__(self, observation_dimensions, size, gamma=0.99, lam=0.95):
        #def __init__(self, observation_dimensions, size, lam=0.95):
        def __init__(self, observation_dimensions, size):
            # Buffer initialization
            self.observation_buffer = np.zeros(
                (size, observation_dimensions), dtype=np.float32
            )
            self.action_buffer = np.zeros(size, dtype=np.int32)
            self.advantage_buffer = np.zeros(size, dtype=np.float32)
            self.reward_buffer = np.zeros(size, dtype=np.float32)
            self.return_buffer = np.zeros(size, dtype=np.float32)
            self.value_buffer = np.zeros(size, dtype=np.float32)
            self.logprobability_buffer = np.zeros(size, dtype=np.float32)
            self.gamma, self.lam = gamma, lam
            self.pointer, self.trajectory_start_index = 0, 0

        def store(self, observation, action, reward, value, logprobability):
            # Append one step of agent-environment interaction
            self.observation_buffer[self.pointer] = observation
            self.action_buffer[self.pointer] = action
            self.reward_buffer[self.pointer] = reward
            self.value_buffer[self.pointer] = value
            self.logprobability_buffer[self.pointer] = logprobability
            self.pointer += 1

        def finish_trajectory(self, last_value=0):
            # Finish the trajectory by computing advantage estimates and rewards-to-go
            path_slice = slice(self.trajectory_start_index, self.pointer)
            rewards = np.append(self.reward_buffer[path_slice], last_value)
            values = np.append(self.value_buffer[path_slice], last_value)

            deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]

            self.advantage_buffer[path_slice] = discounted_cumulative_sums(
                deltas, self.gamma * self.lam
            )
            self.return_buffer[path_slice] = discounted_cumulative_sums(
                rewards, self.gamma
            )[:-1]

            self.trajectory_start_index = self.pointer

        def get(self):
            # Get all data of the buffer and normalize the advantages
            self.pointer, self.trajectory_start_index = 0, 0
            advantage_mean, advantage_std = (
                np.mean(self.advantage_buffer),
                np.std(self.advantage_buffer),
            )
            self.advantage_buffer = (self.advantage_buffer - advantage_mean) / advantage_std
            return (
                self.observation_buffer,
                self.action_buffer,
                self.advantage_buffer,
                self.return_buffer,
                self.logprobability_buffer,
            )
    return Buffer


def test_agent(gamma, lam, clip_ratio, policy_learning_rate, value_function_learning_rate, target_kl):
    Buffer = set_up_class(gamma, lam)

    def mlp(x, sizes, activation=tf.tanh, output_activation=None):
        # Build a feedforward neural network
        for size in sizes[:-1]:
            x = layers.Dense(units=size, activation=activation)(x)
        return layers.Dense(units=sizes[-1], activation=output_activation)(x)


    def logprobabilities(logits, a):
        # Compute the log-probabilities of taking actions a by using the logits (i.e. the output of the actor)
        logprobabilities_all = tf.nn.log_softmax(logits)
        logprobability = tf.reduce_sum(
            tf.one_hot(a, num_actions) * logprobabilities_all, axis=1
        )
        return logprobability


    # Sample action from actor
    @tf.function
    def sample_action(observation):
        logits = actor(observation)
        action = tf.squeeze(tf.random.categorical(logits, 1), axis=1)
        return logits, action


    # Train the policy by maxizing the PPO-Clip objective
    @tf.function
    def train_policy(
        observation_buffer, action_buffer, logprobability_buffer, advantage_buffer
    ):

        with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
            ratio = tf.exp(
                logprobabilities(actor(observation_buffer), action_buffer)
                - logprobability_buffer
            )
            min_advantage = tf.where(
                advantage_buffer > 0,
                (1 + clip_ratio) * advantage_buffer,
                (1 - clip_ratio) * advantage_buffer,
            )

            policy_loss = -tf.reduce_mean(
                tf.minimum(ratio * advantage_buffer, min_advantage)
            )
        policy_grads = tape.gradient(policy_loss, actor.trainable_variables)
        policy_optimizer.apply_gradients(zip(policy_grads, actor.trainable_variables))

        kl = tf.reduce_mean(
            logprobability_buffer
            - logprobabilities(actor(observation_buffer), action_buffer)
        )
        kl = tf.reduce_sum(kl)
        return kl


    # Train the value function by regression on mean-squared error
    @tf.function
    def train_value_function(observation_buffer, return_buffer):
        with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
            value_loss = tf.reduce_mean((return_buffer - critic(observation_buffer)) ** 2)
        value_grads = tape.gradient(value_loss, critic.trainable_variables)
        value_optimizer.apply_gradients(zip(value_grads, critic.trainable_variables))


    """
    ## Hyperparameters
    """

    # Hyperparameters of the PPO algorithm
    steps_per_epoch = 1000
    epochs = 35
    # gamma = 0.99
    #clip_ratio = 0.2
    #policy_learning_rate = 3e-4
    #value_function_learning_rate = 1e-3
    train_policy_iterations = 80
    train_value_iterations = 80
    #lam = 0.97
    #target_kl = 0.01
    hidden_sizes = (64, 64)

    # True if you want to render the environment
    render = False


    """
    ## Initializations
    """

    # Initialize the environment and get the dimensionality of the
    # observation space and the number of possible actions
    env = gym.make("CartPole-v1") #Acrobot-v1 CartPole-v0
    observation_dimensions = env.observation_space.shape[0]
    num_actions = env.action_space.n

    # Initialize the buffer
    buffer = Buffer(observation_dimensions, steps_per_epoch)

    # Initialize the actor and the critic as keras models
    observation_input = keras.Input(shape=(observation_dimensions,), dtype=tf.float32)
    logits = mlp(observation_input, list(hidden_sizes) + [num_actions], tf.tanh, None)
    actor = keras.Model(inputs=observation_input, outputs=logits)
    value = tf.squeeze(
        mlp(observation_input, list(hidden_sizes) + [1], tf.tanh, None), axis=1
    )
    critic = keras.Model(inputs=observation_input, outputs=value)

    # Initialize the policy and the value function optimizers
    policy_optimizer = keras.optimizers.Adam(learning_rate=policy_learning_rate)
    value_optimizer = keras.optimizers.Adam(learning_rate=value_function_learning_rate)

    # Initialize the observation, episode return and episode length
    observation, episode_return, episode_length = env.reset()[0], 0, 0


    """
    ## Train
    """
    rewards = []
    terminal_run = 0
    # Iterate over the number of epochs
    all_episodes = []
    for epoch in range(epochs):
        episodes= []
        # Initialize the sum of the returns, lengths and number of episodes for each epoch
        sum_return = 0
        sum_length = 0
        num_episodes = 0
        reward = 0

        # Iterate over the steps of each epoch
        for t in range(steps_per_epoch):
            if render:
                env.render()

            # Get the logits, action, and take one step in the environment
            observation = observation.reshape(1, -1)
            logits, action = sample_action(observation)
            observation_new, reward, done, _, _ = env.step(action[0].numpy())
            episode_return += reward
            episode_length += 1
            

            # Get the value and log-probability of the action
            value_t = critic(observation)
            logprobability_t = logprobabilities(logits, action)

            # Store obs, act, rew, v_t, logp_pi_t
            buffer.store(observation, action, reward, value_t, logprobability_t)

            # Update the observation
            observation = observation_new

            # Finish trajectory if reached to a terminal state
            terminal = done
            if terminal or (t == steps_per_epoch - 1):
                last_value = 0 if done else critic(observation.reshape(1, -1))
                buffer.finish_trajectory(last_value)
                episodes.append(episode_return) ##################################
                sum_return += episode_return
                sum_length += episode_length
                num_episodes += 1
                observation, episode_return, episode_length = env.reset()[0], 0, 0
        if len(episodes) != 1:
            episodes.remove(episodes[-1]) 


        all_episodes.append(episodes)
        # Get values from the buffer
        (
            observation_buffer,
            action_buffer,
            advantage_buffer,
            return_buffer,
            logprobability_buffer,
        ) = buffer.get()

        # Update the policy and implement early stopping using KL divergence
        for _ in range(train_policy_iterations):
            kl = train_policy(
                observation_buffer, action_buffer, logprobability_buffer, advantage_buffer
            )
            if kl > 1.5 * target_kl:
                # Early Stopping
                break

        # Update the value function
        for _ in range(train_value_iterations):
            train_value_function(observation_buffer, return_buffer)

        # Print mean return and length for each epoch
        rewards.append(sum_return / num_episodes)
        print(
            f" Epoch: {epoch + 1}. Mean Return: {sum_return / num_episodes}.  episode returns: {episodes}"
        )
    print(all_episodes)
    return all_episodes


#data is the list pf 10 agents, each agent rpoduces 10 epoch lists(where an epoch lists is a list of episodes)
#all_data = 

listyy = []
all_data =  []
def test_tuned():
    for i in range(10):
        
        all_episodes = test_agent(0.9997, 0.97, 0.256313, 0.003, 0.001, 0.03) # testing mc params
        # test_agent(0.843372894403312, 0.9599587339508671, 0.2293724995049937, 0.0011442691866330194, 0.0008844199196409004, 0.0768585945177951) cartpole params
        last_10 = all_episodes[-10:]
        all_data.append(all_episodes)
        listyy.append(last_10)
        
        
    data = sum(listyy, [])
    data = sum(data, [])
    
    print('data =' , data, len(data))
    print('all_data = ', all_data)
    return data , all_data

test_tuned()
    # last_n_epochs = 5
    # i = epochs - last_n_epochs
    # loss = sum(rewards[i:])/last_n_epochs

    # for i in range(last_n_epochs, 0, -1):
    #     loss += rewards[epochs - i] - rewards[epochs - (i-1)]


#### for cartpole
    # print('reward = ', loss)
    # return loss
#         if sum_return / num_episodes >= 1000 and terminal_run == 0 and epoch  <= 26:
#             terminal_run = epoch
#             ############### first time reward of 

#         if epoch == terminal_run + 4 and terminal_run != 0:
#             print('reward = ', -1000*terminal_run  + sum(rewards[terminal_run:terminal_run+5])/5)
#             return(-1000*terminal_run  + sum(rewards[terminal_run:terminal_run+5])/5)
#         if epoch == 14 and sum_return / num_episodes < 500:
#             print('reward = ', -3000*epoch)
#             return -3000*epoch

#     print('reward = ', -1000*epochs)
#     return -1000*epochs

# pbounds = {'gamma': (0.8, 0.997), 'lam': (0.95,0.97), 'clip_ratio': (0.1,0.3), 'policy_learning_rate' : (3e-5, 3e-3), 'value_function_learning_rate': (1e-5, 1e-3), 'target_kl': (0.003, 0.3)}

# optimizer = BayesianOptimization(
#     f=test_agent,
#     pbounds=pbounds,
#     random_state=1,
# )

# optimizer.maximize(
#     init_points=10, #2,20
#     n_iter = 100,
# )

# print(optimizer.max)


