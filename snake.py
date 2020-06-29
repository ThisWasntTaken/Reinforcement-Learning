import random
import numpy as np
import torch
from torch.nn import Conv2d, Softmax, MaxPool2d, Module
import gym
from gym import spaces, logger
from gym.utils import seeding
from gym.envs.classic_control import rendering

from algorithms.rainbow_dqn import Rainbow_DQN_Agent
from utils import play, NoisyNormalLinear

class SnakeModel(Module):
    def __init__(self):
        super(SnakeModel, self).__init__()
        self.conv1 = Conv2d(1, out_channels = 16, kernel_size = 5, padding = 2)
        self.maxpool1 = MaxPool2d(2)
        self.conv2 = Conv2d(16, out_channels = 32, kernel_size = 5, padding = 2)
        self.maxpool2 = MaxPool2d(2)
        self.fc1 = NoisyNormalLinear(32 * 5 * 5, 256)
        self.fc2 = NoisyNormalLinear(256, 4)
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = torch.relu(self.conv2(x))
        x = self.maxpool2(x)
        x = x.view(-1, 32 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Snake(gym.Env):
    
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, num_columns = 20, num_rows = 20):
        super(Snake, self).__init__()
        self.num_columns, self.num_rows = num_columns, num_rows
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low = 0, high = 2, shape = (1, self.num_columns, self.num_rows), dtype = np.uint8)
        self.head_x, self.head_y = self.num_columns // 2, self.num_rows // 2
        self.apple_x, self.apple_y = random.randint(0, self.num_columns - 1), random.randint(0, self.num_rows - 1)
        self.eat()
        self.body = [(self.head_x, self.head_y)]
        self.update_state()
        self.viewer = None
    
    def update_state(self):
        self.state = np.full(self.observation_space.shape, -1)
        self.state[0][self.apple_x][self.apple_y] = 1
        for i, j in self.body:
            self.state[0][i][j] = 0
    
    def reset(self):
        self.head_x, self.head_y = self.num_columns // 2, self.num_rows // 2
        self.apple_x, self.apple_y = random.randint(0, self.num_columns - 1), random.randint(0, self.num_rows - 1)
        self.eat()
        self.body = [[self.head_x, self.head_y]]
        self.update_state()
        return self.state
    
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
    
    def length(self):
        return len(self.body)
    
    def render(self, mode = 'human', close = False):
        block_size = 20
        screen_width, screen_height = self.num_columns * block_size, self.num_rows * block_size
        if self.state is None:
            return None
        
        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)
        
        l, r, t, b = self.apple_x * block_size, (self.apple_x + 1) * block_size, self.apple_y * block_size, (self.apple_y + 1) * block_size
        apple = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
        apple.set_color(255, 0, 0)
        apple_trans = rendering.Transform()
        apple.add_attr(apple_trans)
        self.viewer.add_onetime(apple)
        l, r, t, b = self.head_x * block_size, (self.head_x + 1) * block_size, self.head_y * block_size, (self.head_y + 1) * block_size
        cell = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
        cell.set_color(0, 0, 255)
        cell_trans = rendering.Transform()
        cell.add_attr(cell_trans)
        self.viewer.add_onetime(cell)
        for i, j in self.body[: -1]:
            l, r, t, b = i * block_size, (i + 1) * block_size, j * block_size, (j + 1) * block_size
            cell = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            cell.set_color(0, 255, 0)
            cell_trans = rendering.Transform()
            cell.add_attr(cell_trans)
            self.viewer.add_onetime(cell)

        return self.viewer.render(return_rgb_array = (mode == 'rgb_array'))
    
    def eat(self):
        if self.head_x == self.apple_x and self.head_y == self.apple_y:
            while [self.apple_x, self.apple_y] in self.body:
                self.apple_x, self.apple_y = random.randint(0, self.num_columns - 1), random.randint(0, self.num_rows - 1)
            return 1, True
        return 0, False
    
    def crash(self):
        if self.head_x >= self.num_columns or self.head_x < 0\
        or self.head_y >= self.num_rows or self.head_y < 0\
        or [self.head_x, self.head_y] in self.body[:-1]:
            return True
        return False
    
    def step(self, action):
        reward = 0
        actions = {0 : 'LEFT', 1 : 'RIGHT', 2 : 'UP', 3 : 'DOWN'}
        x_change, y_change = 0, 0
        if actions[action] == 'LEFT'    : x_change, y_change = -1, 0
        elif actions[action] == 'RIGHT' : x_change, y_change = 1, 0
        elif actions[action] == 'UP'    : x_change, y_change = 0, -1
        elif actions[action] == 'DOWN'  : x_change, y_change = 0, 1

        self.head_x += x_change
        self.head_y += y_change
        self.body.append([self.head_x, self.head_y])
        if self.crash():
            return self.state, -1, True, {}
        else:
            reward, ate = self.eat()
            if not ate: self.body.pop(0)
        
        self.update_state()
        return self.state, reward, False, {}

if __name__ == "__main__":
    env = Snake()
    agent = Rainbow_DQN_Agent(
        environment = env,
        model_class = SnakeModel,
        learning_rate = 0.001,
        gamma = 0.99,
        replay_buffer_size = 20000,
        minimum_buffer_size = 5000,
        noisy_net = True,
        epsilon = 1,
        epsilon_decay = 0.999,
        epsilon_min = 0.01,
        prioritized_sample = True,
        alpha = 0.5,
        beta0 = 0.6,
        beta_iters = 100000,
        transfer_frequency = 1000,
        device = torch.device('cuda:0')
    )
    agent.train(
        num_episodes = 2000,
        save_path = 'models/snake_model.pth',
        batch_size = 128
    )
    play(
        environment = env,
        model_class = SnakeModel,
        model_path = 'models/snake_model.pth',
        num_episodes = 1
    )