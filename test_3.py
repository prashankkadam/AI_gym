# In this code we will balance the CartPole based using a trained neural
# network

# Importing the required libraries
import gym
import random
import numpy as np
import tensorflow as tf
from collections import deque

# Printing the current version of AIGym and TensorFlow
print("Gym:", gym.__version__)
print("Tensorflow:", tf.__version__)

# Loading the cartpole environment in the gym
env_name = "CartPole-v0"
env = gym.make(env_name)


# Creating a class for designing, updating and getting the q state
# of the neural network
# Here we create a two dense layered neural network with ReLU
# activation and Adam Optimizer
# The model is updated using the update_model method where the state,
# action and the q_target to the nn are updated
class QNetwork():
    def __init__(self, state_dim, action_size):
        self.state_in = tf.placeholder(tf.float32,
                                       shape=[None, *state_dim])
        self.action_in = tf.placeholder(tf.int32, shape=[None])
        self.q_target_in = tf.placeholder(tf.float32,
                                          shape=[None])
        action_one_hot = tf.one_hot(self.action_in,
                                    depth=action_size)
        self.hidden1 = tf.layers.dense(self.state_in,
                                       100,
                                       activation=tf.nn.relu)
        self.q_state = tf.layers.dense(self.hidden1,
                                       action_size,
                                       activation=None)
        self.q_state_action = tf.reduce_sum(tf.multiply(self.q_state,
                                                        action_one_hot),
                                            axis=1)
        self.loss = tf.reduce_mean(tf.square(self.q_state_action -
                                             self.q_target_in))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)

    def update_model(self, session, state, action, q_target):
        feed = {self.state_in: state,
                self.action_in: action,
                self.q_target_in: q_target}
        session.run(self.optimizer, feed_dict=feed)

    def get_q_state(self, session, state):
        q_state = session.run(self.q_state,
                              feed_dict={self.state_in: state})
        return q_state


# Here we define a class to store optimum values of the training in
# buffer memory, these values can then be used to speed up the
# training process
# The add and sample methods are used to perform append and subset
# operations on the replay buffer
class ReplayBuffer():
    def __init__(self, maxlen):
        self.buffer = deque(maxlen=maxlen)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        sample_size = min(len(self.buffer), batch_size)
        samples = random.choices(self.buffer, k=sample_size)
        return map(list, zip(*samples))


# In this class we write the function to fetch the action to be
# performed by the cartpole and also to train the nn
# get_action is used to find the action suitable for the nn to
# take and train is used to update the hyper parameters of the
# model in each epoch
class DQNAgent():
    def __init__(self, env):
        self.state_dim = env.observation_space.shape
        self.action_size = env.action_space.n
        self.q_network = QNetwork(self.state_dim,
                                  self.action_size)
        self.replay_buffer = ReplayBuffer(maxlen=10000)
        self.gamma = 0.97
        self.eps = 1.0

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def get_action(self, state):
        q_state = self.q_network.get_q_state(self.sess,
                                             [state])
        action_greedy = np.argmax(q_state)
        # action = action_greedy
        action_random = np.random.randint(self.action_size)
        action = action_random if random.random() < self.eps else action_greedy
        return action

    def train(self, state, action, next_state, reward, done):
        self.replay_buffer.add((state,
                                action,
                                next_state,
                                reward,
                                done))
        states, actions, next_states, rewards, dones = self.replay_buffer.sample(50)
        q_next_states = self.q_network.get_q_state(self.sess,
                                                   next_states)
        q_next_states[dones] = np.zeros([self.action_size])
        q_targets = rewards + self.gamma * np.max(q_next_states,
                                                  axis=1)
        self.q_network.update_model(self.sess,
                                    states,
                                    actions,
                                    q_targets)

        if done: self.eps = max(0.1, 0.99 * self.eps)

    def __del__(self):
        self.sess.close()


# Creating an object of the DQNAgent class
agent = DQNAgent(env)
# number of training cycles
num_episodes = 400

# Calling all our defined function to train the nn for the
# defined number of cycles and printing the score after each
# cycle
for ep in range(num_episodes):
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        action = agent.get_action(state)
        next_state, reward, done, info = env.step(action)
        agent.train(state, action, next_state, reward, done)
        env.render()
        total_reward += reward
        state = next_state

    print("Episode: {}, total_reward: {:.2f}".format(ep,
                                                     total_reward))
