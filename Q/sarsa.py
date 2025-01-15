#Implimenting sarsa

import gymnasium as gym
import numpy as np
import math

class CartPoleAgent():
    #original (self, buckets=(1, 1, 6, 12), num_episodes=1000, min_lr=0.1, min_epsilon=0.1, discount=0.98, decay=50)
    def __init__(self, buckets=(1, 1, 6, 12), num_episodes=1000, min_lr=0.1, min_epsilon=0.1, discount=0.98, decay=50):
        self.buckets = buckets
        self.num_episodes = num_episodes
        self.min_lr = min_lr
        self.min_epsilon = min_epsilon
        self.discount = discount
        self.decay = decay

        self.env = gym.make('CartPole-v1')

        # [position, velocity, angle, angular velocity]
        self.upper_bounds = [self.env.observation_space.high[0], 0.5, self.env.observation_space.high[2], math.radians(50) / 1.]
        self.lower_bounds = [self.env.observation_space.low[0], -0.5, self.env.observation_space.low[2], -math.radians(50) / 1.]

        self.sarsa_table = np.zeros(self.buckets + (self.env.action_space.n,))
        
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
            return np.argmax(self.sarsa_table[state])

    def update_sarsa(self, state, action, reward, new_state, new_action):
        self.sarsa_table[state][action] += self.learning_rate * (reward + self.discount * (self.sarsa_table[new_state][new_action]) - self.sarsa_table[state][action])

    def get_epsilon(self, t):
        return max(self.min_epsilon, min(1., 1. - math.log10((t + 1) / self.decay)))

    def get_learning_rate(self, t):
        return max(self.min_lr, min(1., 1. - math.log10((t + 1) / self.decay)))

    def train(self, eps):
        for e in range(eps):
            current_state = self.discretize_state(self.env.reset())

            if e % 100 == 0:
                print(f'Episode : {e}')

            self.learning_rate = self.get_learning_rate(e)
            self.epsilon = self.get_epsilon(e)
            done = False

            while not done:
                action = self.choose_action(current_state)
                obs, reward, done, _1, _2 = self.env.step(action)
                new_state = self.discretize_state(obs)
                new_action = self.choose_action(new_state)
                self.update_sarsa(current_state, action, reward, new_state, new_action)
                current_state = new_state

        print('Finished training!')

    def run(self):
        # self.env = gym.make("CartPole-v1", render_mode="human")
        self.env = gym.make("CartPole-v1")
        t = 0
        done = False
        current_state = self.discretize_state(self.env.reset())
        while not done:
                self.env.render()
                t = t+1
                action = self.choose_action(current_state)
                obs, reward, done, _1, _2 = self.env.step(action)
                new_state = self.discretize_state(obs)
                current_state = new_state
                # if t == 195:
                #     return t
                
        return t   
            

ave_over_20_test = []
max_over_20_test = []

for k in range(10):
    eps = (k+21)*20
    
    for i in range(100):
        if __name__ == "__main__":
            agent = CartPoleAgent()
            agent.train(eps)
            scores = []
            for j in range(20):
                t = agent.run()
                # print(f"episodes trained on: {eps} agent: {i} episode: {j} score: {t}")
                scores.append(t)

            #Get average and max of the test  
            ave_over_20_test.append(sum(scores)/20)
            max_over_20_test.append(max(scores))
            print('Eps = ', eps, ', agent = ', i, ': ave = ', sum(scores)/20, ' | max = ', max(scores))

print(ave_over_20_test)
print(max_over_20_test)