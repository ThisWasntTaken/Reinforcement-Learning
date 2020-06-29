import math
import torch
from torch.nn import functional as F, Module, init
from torch.nn.parameter import Parameter

class NoisyNormalLinear(Module):
    def __init__(self, in_features, out_features, mean=0, std=1, bias=True):
        super(NoisyNormalLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer('mean', torch.tensor(mean))
        self.register_buffer('std', torch.tensor(std))
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.weight_noise = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
            self.bias_noise = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
            self.register_parameter('bias_noise', None)
        self.reset_parameters()
    
    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        init.normal_(self.weight_noise, self.mean, self.std)
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
            init.normal_(self.bias_noise, self.mean, self.std)
    
    def forward(self, input):
        if not self.training:
            return F.linear(input, self.weight, self.bias)
        return F.linear(
            input,
            self.weight + self.weight_noise,
            (self.bias + self.bias_noise) if self.bias is not None else self.bias
        )
    
    def extra_repr(self):
        return 'in_features={}, out_features={}, mean={}, std={}, bias={}'.format(
            self.in_features, self.out_features, self.mean, self.std, self.bias is not None
        )
    
def play(environment, model_class, model_path, num_episodes = 1):
    model = model_class()
    model.load_state_dict(torch.load(model_path))
    model.training = False
    for _ in range(num_episodes):
        t = 0
        done = False
        state = environment.reset()
        while not done:
            environment.render()
            t += 1
            inp = torch.from_numpy(state)
            inp = torch.reshape(inp, [1] + list(environment.observation_space.shape))
            inp = inp.float()
            action = torch.argmax(model.forward(inp)).item()
            state, _, done, _ = environment.step(action)
        print("Done at step {}".format(t))
    environment.close()