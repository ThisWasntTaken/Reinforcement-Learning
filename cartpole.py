import gym
import torch
from torch.nn import Conv2d, Linear, Softmax, MaxPool2d, Module

from algorithms.vanilla_dqn import Vanilla_DQN_Agent
from algorithms.rainbow_dqn import Rainbow_DQN_Agent
from utils import play, NoisyNormalLinear

class CartPoleModel(Module):
    def __init__(self):
        super(CartPoleModel, self).__init__()
        self.fc1 = NoisyNormalLinear(4, 24)
        self.fc2 = NoisyNormalLinear(24, 24)
        self.fc3 = NoisyNormalLinear(24, 2)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    agent = Vanilla_DQN_Agent(
        environment = env,
        model_class = CartPoleModel,
        learning_rate = 0.01,
        gamma = 0.95,
        epsilon = 1,
        epsilon_decay = 0.999,
        epsilon_min = 0.01,
        replay_buffer_size = 10000,
        minimum_buffer_size = 1000,
        transfer_frequency = 500,
        device = torch.device('cpu')
    )
    # agent = Rainbow_DQN_Agent(
    #     environment = env,
    #     model_class = CartPoleModel,
    #     learning_rate = 0.01,
    #     gamma = 0.95,
    #     replay_buffer_size = 10000,
    #     minimum_buffer_size = 1000,
    #     epsilon = 1,
    #     epsilon_decay = 0.999,
    #     epsilon_min = 0.01,
    #     prioritized_sample = False,
    #     transfer_frequency = 500,
    #     device = torch.device('cpu')
    # )
    agent.train(
        num_episodes = 800,
        save_path = 'models/model.pth',
        batch_size = 128
    )
    play(
        environment = env,
        model_class = CartPoleModel,
        model_path = 'models/model.pth',
        num_episodes = 1
    )