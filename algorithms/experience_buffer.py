import random
import numpy as np
import torch

class UniformExperienceReplayBuffer:
    def __init__(self, capacity, state_shape):
        super(UniformExperienceReplayBuffer, self).__init__()
        self.capacity = capacity
        self.states = np.zeros(([capacity] + state_shape))
        self.actions = np.zeros((capacity))
        self.new_states = np.zeros(([capacity] + state_shape))
        self.rewards = torch.zeros((capacity))
        self.mask = torch.ones((capacity))
        self.position = 0
        self.size = 0
    
    def push(self, experience):
        self.states[self.position] = experience[0]
        self.actions[self.position] = experience[1]
        self.rewards[self.position] = experience[3]
        if experience[2] is None:
            self.mask[self.position] = 0
            self.new_states[self.position] = 0
        else:
            self.mask[self.position] = 1
            self.new_states[self.position] = experience[2]
            
        self.position = (self.position + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1
    
    def sample(self, batch_size):
        indices = random.choices(
            population = range(self.size),
            k = batch_size
        )
        return self.states[indices],\
            self.actions[indices],\
            self.new_states[indices],\
            self.rewards[indices],\
            self.mask[indices]
    
    def __len__(self):
        return self.size if self.size < self.capacity else self.capacity

class PrioritizedExperienceReplayBuffer(UniformExperienceReplayBuffer):
    def __init__(self, capacity, state_shape, alpha, beta0, beta_iters):
        super(PrioritizedExperienceReplayBuffer, self).__init__(capacity, state_shape)
        self.priorities = torch.zeros((capacity))
        self.alpha = alpha
        self.beta = beta0
        self.beta_incr = ((1 - beta0) / beta_iters) if beta_iters else 0
        self.max_priority = 0
    
    def push(self, experience):
        if self.priorities[self.position] == 0:
            self.priorities[self.position] = 1
            self.max_priority = 1
        else:
            self.priorities[self.position] = self.max_priority
        super(PrioritizedExperienceReplayBuffer, self).push(experience)
    
    def sample(self, batch_size):
        priorities_sum = self.priorities[:self.size].pow(self.alpha).sum()
        probabilties = self.priorities[:self.size].pow(self.alpha) / priorities_sum
        indices = random.choices(
            population = range(self.size),
            weights = probabilties,
            k = batch_size
        )
        self.beta += self.beta_incr
        importance_weights = (self.size * probabilties[indices]).pow(-self.beta)
        importance_weights = importance_weights / torch.max(importance_weights)
        return self.states[indices],\
            self.actions[indices],\
            self.new_states[indices],\
            self.rewards[indices],\
            self.mask[indices],\
            importance_weights,\
            indices
    
    def update(self, priorities, indices):
        self.priorities[indices] = torch.abs(priorities) + 1e-5
        self.max_priority = max(self.max_priority, max(priorities))