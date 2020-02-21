# In this code, we try to balance the cartpole based on the pole angle
# if the pole angle is less than 0 we go left else we go right
# Importing the required packages
import gym

# Setting the environment name
env_name = "CartPole-v1"
# Creating an environment
env = gym.make(env_name)

# Printing the structure of the observation table and the action space
print(env.observation_space)
print(env.action_space)


# Now we create a class to define various function that need to be called
# when a certain action is required to be taken on the cartpole
class Agent():
    # We initialize the constructor and save the action size as the total
    # number of actions available
    def __init__(self, env):
        self.action_size = env.action_space.n

    # This method return the action to be performed on the cartpole
    def get_action(self, state):
        # We can retrieve the pole angle from the second position of the
        # state vector
        pole_angle = state[2]
        # Applying the logic for pole movement based on the pole angle
        action = 0 if pole_angle < 0 else 1
        return action


# Creating object of Agent class
agent = Agent(env)
# This function returns the current state of the environment
state = env.reset()

# Running our code for 200 actions
for _ in range(200):
    action = agent.get_action(state)
    state, reward, done, info = env.step(action)

    # Rendering the final result
    env.render()
