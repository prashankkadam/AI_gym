# In this code we use the Hill climbing algorithm to balance the
# CartPole. Hill climbing algorithms basically generates an intial
# random matrix and it's values are provided as an input to the
# algorithm. The algorithm tries to find the best set of combinations
# for the CartPole such that it balances with +-12 degrees

# Importing the required libraries
import gym
import numpy as np

# Creating the CartPole environment
env_name = "CartPole-v0"
env = gym.make(env_name)


# Creating a class for defining our model, fetching the required action
# and updating the model
class HillClimbingAgent():
    def __init__(self, env):
        self.state_dim = env.observation_space.shape
        self.action_size = env.action_space.n
        self.build_model()

    def build_model(self):
        self.weights = 1e-4*np.random.rand(*self.state_dim,
                                           self.action_size)
        self.best_reward = -np.Inf
        self.best_weights = np.copy(self.weights)
        self.noise_scale = 1e-2

    def get_action(self, state):
        p = np.dot(state, self.weights)
        action = np.argmax(p)
        return action

    def update_model(self, reward):
        if reward >= self.best_reward:
            self.best_reward = reward
            self.best_weights = np.copy(self.weights)
            self.noise_scale = max(self.noise_scale/2, 1e-3)
        else:
            self.noise_scale = min(self.noise_scale*2, 2)

        self.weights= self.best_weights + self.noise_scale * np.random.rand(*self.state_dim,
                                                                            self.action_size)


# Creating the object for the above created class
agent = HillClimbingAgent(env)
# Defining the number of iterations our algorithms has to go through
num_episodes = 100

# Looping over our algorithm for the required number of iterations and
# updating the model accordingly. If the current reward is better that the
# previous reward, we swap the best weights with the current weights else
# we keep the weight and go to the next random variable. Finally we print
# the total reward for each iteration
for ep in range(num_episodes):
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        action = agent.get_action(state)
        state, reward, done, info = env.step(action)
        env.render()
        total_reward += reward

    agent.update_model(total_reward)
    print("Episode: {}, total_reward: {:.2f}".format(ep,
                                                     total_reward))
