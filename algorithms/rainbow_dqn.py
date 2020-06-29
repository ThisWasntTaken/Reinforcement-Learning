import random
import numpy as np
import torch
from torch.optim import SGD, Adam
import tqdm

from .experience_buffer import UniformExperienceReplayBuffer, PrioritizedExperienceReplayBuffer

class Rainbow_DQN_Agent:
    def __init__(self, environment, model_class, learning_rate, gamma = 0.9,
            replay_buffer_size = 5000, minimum_buffer_size = 1000, noisy_net = False,
            epsilon = 1, epsilon_decay = 0.999, epsilon_min = 0.01,
            prioritized_sample = False, alpha = 0, beta0 = 0, beta_iters = None,
            transfer_frequency = 0, device = torch.device('cpu')
        ):
        super(Rainbow_DQN_Agent, self).__init__()
        assert (minimum_buffer_size < replay_buffer_size),\
            "minimum_buffer_size should not be less than replay_buffer_size."
        assert not (prioritized_sample and alpha == 0),\
            "Prioritized sampling requires positive alpha."
        self.env = environment
        self.state_shape = list(self.env.observation_space.shape)
        self.prioritized_sample = prioritized_sample
        if not self.prioritized_sample:
            self.replay_buffer = UniformExperienceReplayBuffer(replay_buffer_size, self.state_shape)
        else:
            self.replay_buffer = PrioritizedExperienceReplayBuffer(replay_buffer_size, self.state_shape, alpha, beta0, beta_iters)
        self.minimum_buffer_size = minimum_buffer_size
        self.noisy_net = noisy_net
        if noisy_net:
            self.epsilon = 0
            self.epsilon_decay = 0
            self.epsilon_min = 0
        else:
            self.epsilon = epsilon
            self.epsilon_decay = epsilon_decay
            self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.device = device
        self.training_model = model_class().to(self.device)
        self.optimizer = Adam(self.training_model.parameters(), lr = learning_rate)
        self.target_model = model_class().to(self.device)
        self.target_model.training = False
        self.target_model.load_state_dict(self.training_model.state_dict())
        self.transfer_frequency = transfer_frequency
    
    def get_action(self, state):
        gen = random.uniform(0, 1)
        if gen >= self.epsilon:
            inp = torch.from_numpy(state.reshape([1] + self.state_shape))
            inp = inp.to(self.device).float()
            q_values = self.training_model.forward(inp)
            return torch.argmax(q_values)
        else:
            return torch.tensor(self.env.action_space.sample())
    
    def get_best_actions(self, states):
        inp = torch.from_numpy(states)
        inp = inp.to(self.device).float()
        q_values = self.training_model.forward(inp)
        return torch.argmax(q_values, axis = 1).detach().cpu().numpy()
    
    def get_training_q_value(self, states, actions):
        inp = torch.from_numpy(states)
        inp = inp.to(self.device).float()
        q_values = self.training_model.forward(inp)
        return q_values[range(q_values.shape[0]), actions]
    
    def get_target_q_value(self, states, actions):
        inp = torch.from_numpy(states)
        inp = inp.to(self.device).float()
        q_values = self.target_model.forward(inp)
        return q_values[range(q_values.shape[0]), actions]
    
    def train(self, num_episodes, save_path, batch_size):
        loss_file = open("results/losses.txt", "w")
        reward_file = open("results/rewards.txt", "w")
        
        step = 0
        for e in tqdm.tqdm(range(1, num_episodes + 1)):
            state = self.env.reset()
            done = False
            episode_reward = 0
            if self.prioritized_sample:
                gradient_accum = [torch.zeros(param.shape).to(self.device) for param in self.training_model.parameters()]
            while not done:
                step += 1
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
                action = self.get_action(state)
                # self.env.render()
                new_state, reward, done, _ = self.env.step(action.item())
                self.replay_buffer.push([state, action, new_state if not done else None, reward])
                episode_reward += reward
                state = new_state

                if len(self.replay_buffer) > self.minimum_buffer_size:
                    if self.prioritized_sample:
                        states, actions, new_states, rewards, mask, importance_weights, indices = self.replay_buffer.sample(batch_size)
                        importance_weights = importance_weights.to(self.device)
                    else:
                        states, actions, new_states, rewards, mask = self.replay_buffer.sample(batch_size)
                    
                    rewards = rewards.to(self.device)
                    mask = mask.to(self.device)
                    best_actions = self.get_best_actions(new_states)
                    target_q_values = self.get_target_q_value(new_states, best_actions) * mask
                    expected_q_values = self.get_training_q_value(states, actions)
                    errors = (
                        rewards + self.gamma * target_q_values - expected_q_values
                    ).pow(2)

                    if self.prioritized_sample:
                        self.replay_buffer.update(errors.detach().cpu(), indices)

                    loss = errors.mean().sqrt()
                    loss_file.write("{}\n".format(loss.item()))
                    self.optimizer.zero_grad()
                    loss.backward()
                    
                    if self.prioritized_sample:
                        with torch.no_grad():
                            for i, param in enumerate(self.training_model.parameters()):
                                gradient_accum[i] += torch.dot(importance_weights, errors) * param.grad

                    with torch.no_grad():
                        for param in self.training_model.parameters():
                            param.grad.data.clamp_(-1, 1)

                    self.optimizer.step()

                    if (self.transfer_frequency > 0) and (step % self.transfer_frequency == 0):
                        self.target_model.load_state_dict(self.training_model.state_dict())
            
            if self.prioritized_sample:
                with torch.no_grad():
                    for i, param in enumerate(self.training_model.parameters()):
                        new_param = param + gradient_accum[i]
                        param.data.copy_(new_param)
            
            reward_file.write("{}\t{}\n".format(e, episode_reward))
        
        self.target_model.load_state_dict(self.training_model.state_dict())
        torch.save(self.target_model.state_dict(), save_path)
        loss_file.close()
        reward_file.close()
        self.env.close()