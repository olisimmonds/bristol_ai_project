#Implimenting q-learning for cartpole

import gymnasium as gym
import numpy as np
import math
import statistics
import random

from bayes_opt import BayesianOptimization

def set_up_agent(buckets):
    class CartPoleAgent():
        #original (self, buckets=(1, 1, 6, 12), num_episodes=1000, min_lr=0.1, min_epsilon=0.1, discount=0.98, decay=25)
        # def __init__(self, buckets=(1, 1, 6, 12), num_episodes=40, min_lr=0.1, min_epsilon=0.1, discount=0.98, decay=50):
        #     self.buckets = buckets
        #     self.num_episodes = num_episodes
        #     self.min_lr = min_lr
        #     self.min_epsilon = min_epsilon
        #     self.discount = discount
        #     self.decay = decay

        def __init__(self, num_episodes=1000):
            self.buckets = buckets
            self.num_episodes = num_episodes
            # self.min_lr = min_lr
            # self.min_epsilon = min_epsilon
            # self.discount = discount
            # self.decay = decay

            self.env = gym.make('CartPole-v1')

            # [position, velocity, angle, angular velocity]
            self.upper_bounds = [self.env.observation_space.high[0], 0.5, self.env.observation_space.high[2], math.radians(50) / 1.]
            self.lower_bounds = [self.env.observation_space.low[0], -0.5, self.env.observation_space.low[2], -math.radians(50) / 1.]

            self.q_learning_table = np.zeros(self.buckets + (self.env.action_space.n,))
            
        def discretize_state(self, obs):
            discretized = list()
            if type(obs) == tuple:
                obs = obs[0]
            for i in range(len(obs)):
                scaling = (obs[i] + abs(self.lower_bounds[i])) / (self.upper_bounds[i] - self.lower_bounds[i])
                new_obs = int(round((self.buckets[i] - 1) * scaling))
                new_obs = min(self.buckets[i] - 1, max(0, new_obs))
                discretized.append(new_obs)
            return tuple(discretized)

        def choose_action(self, state):
            if (np.random.random() < self.epsilon):
                return self.env.action_space.sample() 
            else:
                return np.argmax(self.q_learning_table[state])

        def update_q_learning(self, state, action, reward, next_state, discount):
            """
            Updates the Q-table with the Q-learning algorithm.
            """
            old_q_value = self.q_learning_table[state][action]
            next_max_q_value = np.max(self.q_learning_table[next_state])
            new_q_value = (1 - self.learning_rate) * old_q_value + self.learning_rate * (reward + discount * next_max_q_value)
            self.q_learning_table[state][action] = new_q_value

        def get_epsilon(self, t, min_epsilon, decay):
            return max(min_epsilon, min(1., 1. - math.log10((t + 1) / decay)))

        def get_learning_rate(self, t, min_lr, decay):
            return max(min_lr, min(1., 1. - math.log10((t + 1) / decay)))

        def train(self, min_lr, min_epsilon, discount, decay):
            for e in range(self.num_episodes):
                current_state = self.discretize_state(self.env.reset()[0])

                # if e % 1000 == 0:
                #     print(f'Episode : {e}')

                self.learning_rate = self.get_learning_rate(e, min_lr, decay)
                self.epsilon = self.get_epsilon(e, min_epsilon, decay)

                done = False

                while not done:
                    action = self.choose_action(current_state)
                    obs, reward, done, _1, _2 = self.env.step(action)
                    new_state = self.discretize_state(obs)
                    self.update_q_learning(current_state, action, reward, new_state, discount)
                    current_state = new_state

            print('Finished training!')

        def run(self):
            # self.env = gym.make("CartPole-v1", render_mode="human")
            self.env = gym.make("CartPole-v1")
            score = 0
            done = False
            current_state = self.discretize_state(self.env.reset()[0])
            while not done and score < 500:
                    self.env.render()
                    action = np.argmax(self.q_learning_table[current_state])
                    obs, reward, done, _1, _2 = self.env.step(action)
                    new_state = self.discretize_state(obs)
                    current_state = new_state
                    score += reward
            return score
    return CartPoleAgent()


# Explain why i use ave instead of max
def test_agent(min_lr, min_epsilon, discount, decay, bucket_0, bucket_1, bucket_2, bucket_3):
    buckets = tuple([int(bucket_0), int(bucket_1), int(bucket_2), int(bucket_3)]) 
    scores = []
    #test 5 agents
    for i in range(1):
        if __name__ == "__main__":    
            agent = set_up_agent(buckets)      
            # agent = CartPoleAgent()
            agent.train(min_lr, min_epsilon, discount, decay)
            
            #test each agent 100 times
            for j in range(100):
                t = agent.run()
                scores.append(t)

            #Get average of the test    
    ave_of_agent = sum(scores)/100

    return ave_of_agent
            

#Bayes opt

# pbounds = {'min_lr': (0.001, 0.5), 'min_epsilon': (0.001, 0.5), 'discount': (0.1, 0.9999), 'decay': (1, 200), 'bucket_0': (1, 30), 'bucket_1': (1, 30), 'bucket_2': (1, 30), 'bucket_3': (1, 30)}

# optimizer = BayesianOptimization(
#     f=test_agent,
#     pbounds=pbounds,
#     random_state=1,
# )

# optimizer.maximize(
#     init_points=50,
#     n_iter=500,
# )

# print(optimizer.max)

# Opt params for cartpole
# {'target': 20366.82, 'params': {'bucket_0': 1.1544269302452057, 'bucket_1': 11.876771802211112, 'bucket_2': 29.078271526727637, 'bucket_3': 11.55171329112111, 'decay': 69.67505733621513, 'discount': 0.9560266938018456, 'min_epsilon': 0.06306044698237305, 'min_lr': 0.38732364823894777}}

# Opt params for mountain car
# {'target': -133.57, 'params': {'bucket_0': 26.58047026722138, 'bucket_1': 21.66550004615175, 'decay': 111.67543600719513, 'discount': 0.5601072581830205, 'min_epsilon': 0.41041559197591426, 'min_lr': 0.04020428828777738}}

# Opt params for acrobat
# {'target': -98.01, 'params': {'bucket_0': 9.99557425586167, 'bucket_1': 1.0, 'bucket_2': 1.0, 'bucket_3': 1.0, 'bucket_4': 19.616196302060988, 'bucket_5': 30.0, 'decay': 42.14476816745908, 'discount': 0.9999, 'min_epsilon': 0.5, 'min_lr': 0.001}}


# Test using opt buckets from cartpole opt
# Will test_agent on each set of params 100 times
params_opt_on_cartpole = []
params_opt_on_mountaincar = []
params_opt_on_acrobat = []

trials = 100
for i in range(trials):

    cartpole = test_agent(0.38732364823894777, 0.06306044698237305, 0.9560266938018456, 69.67505733621513, 1.1544269302452057, 11.876771802211112, 29.078271526727637, 11.55171329112111)
    mountain_car = test_agent(0.04020428828777738, 0.41041559197591426, 0.5601072581830205, 111.67543600719513, 1.1544269302452057, 11.876771802211112, 29.078271526727637, 11.55171329112111)
    acrobat = test_agent(0.001, 0.5, 0.9999, 42.14476816745908, 1.1544269302452057, 11.876771802211112, 29.078271526727637, 11.55171329112111)

    print('trial = ', i)
    print('Using opt for cartpole', cartpole)
    print('Using opt for mountain car', mountain_car)
    print('Using opt for acrobat', acrobat)

    params_opt_on_cartpole.append(cartpole)
    params_opt_on_mountaincar.append(mountain_car)
    params_opt_on_acrobat.append(acrobat)

print('rewards for agent trained on cartpole params = ', params_opt_on_cartpole)
print('rewards for agent trained on mountaincar params = ', params_opt_on_mountaincar)
print('rewards for agent trained on acrobat params = ', params_opt_on_acrobat)



#Random optermisation
# Init min performance and params
# best_performance = 0
# best_params = {'num_episodes': 0, 'min_lr': 0, 'min_epsilon': 0, 'discount': 0, 'decay': 0}
# # Performing random search hyperparameter tuning
# for k in range(100):

#     #Explain why I have chosen these ranges

#     # num_episodes = random multiple of 50 between 50 and 1000 
#     num_episodes = random.randint(1,20)*50
#     # min_lr and min_epsilon are random multiples of 0.1 between 0.1 and 0.5
#     min_lr = random.randint(1,5)*0.1
#     min_epsilon = random.randint(1,5)*0.1
#     # discount is random multiple of 0.1 be`0
#     # .tween 0.5 and 0.9
#     discount = random.randint(5,9)*0.1
#     # decay is random multiple of 20 between 20 and 100
#     decay = random.randint(1,5)*20

#     print('test = ', k)
#     test = test_agent()
#     # If we have an improved agent we update best parameters
#     if test > best_performance:

#         best_performance = test
#         best_params['num_episodes'] = num_episodes
#         best_params['min_lr'] = min_lr
#         best_params['min_epsilon'] = min_epsilon
#         best_params['discount'] = discount
#         best_params['decay'] = decay

#         print(test)
#         print('num_episodes = ', num_episodes, ' | min_lr = ', min_lr, ' | min_epsilon = ', min_epsilon, ' | discount = ', discount, ' | decay = ', decay)


# print(best_params)
# print(best_performance)


# Tested on cartpole
# ave reward for agent trained on cartpole params =  1734.8162000000002
# ave reward for agent trained on mountaincar params =  252.71419999999992
# ave reward for agent trained on acrobat params =  462.84079999999994