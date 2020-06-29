import gym
import torch
from torch.nn import Conv2d, Linear, Softmax, MaxPool2d, Module

from algorithms.rainbow_dqn_agent import Rainbow_DQN_Agent
from utils import play, NoisyNormalLinear

class CartPoleModel(Module):
    def __init__(self):
        super(CartPoleModel, self).__init__()
        self.fc1 = NoisyNormalLinear(2, 24)
        self.fc2 = NoisyNormalLinear(24, 24)
        self.fc3 = NoisyNormalLinear(24, 3)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == "__main__":
    env = gym.make('MountainCar-v0')
    agent = Rainbow_DQN_Agent(
        environment = env,
        model_class = CartPoleModel,
        learning_rate = 0.01,
        gamma = 0.95,
        replay_buffer_size = 10000,
        minimum_buffer_size = 1000,
        prioritized_sample = False,
        alpha = 0,
        beta0 = 0,
        beta_iters = 0,
        transfer_frequency = 200,
        device = torch.device('cpu')
    )
    agent.train(
        num_episodes = 500,
        save_path = 'models/mountaincar_model.pth',
        batch_size = 128
    )
    play(
        environment = env,
        model_class = CartPoleModel,
        model_path = 'models/mountaincar_model.pth',
        num_episodes = 1
    )