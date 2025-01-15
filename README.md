This is the code from a project completed in collaboration with four other students for the module Introduction to Artificial Intelligence during my BSc in Mathematics at the University of Bristol. It received a grade of 85%. The code consists of various AI models competing in games in the OpenAI Gym. For methods and results please see https://github.com/olisimmonds/bristol_ai_project/blob/3f49f082492f51cdf02ef54978020a806dfe3ae7/Intro_to_AI_Project_write_up.pdf. See the abstract below.

Abstract—This report explores the generalisability of modern
reinforcement learning (RL) algorithms using the OpenAI Gym
environments. The core of our methodology consisted of tuning
an algorithm in one environment and assessing its performance
across domains, utilising an adaptation of the ‘inter-algorithm
normalisation’ method to standardise testing. We adapted Qlearning to ensure compatibility in continuous environments as
well as created a new loss function when tuning the Proximal
Policy Optimisation (PPO) algorithm to optimise the trade-off
between training time and performance. Our findings show that
Q-Learning successfully generalised across simple domains, but
failed in the more complex environment LunarLander. Extending
the algorithm via a neural network (NN) to a Deep Q Network
(DQN) remediated this failure. In contrast, the PPO algorithm
performed well across domains, but tended to over-fit, leading to
the DQN being the most generalisable algorithm we investigated.

