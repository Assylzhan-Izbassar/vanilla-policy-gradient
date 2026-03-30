import gymnasium as gym


def train(env_name, hidden_size, lr, epochs, batch_size):
    env = gym.make(env_name)
