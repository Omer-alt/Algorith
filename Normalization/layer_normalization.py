import torch 
from torch import nn

class LayerNormalization:
    def __init__(self, parameters_shape, eps=1e-5):
        self.parameters_shape = parameters_shape
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(parameters_shape))
        self.beta = nn.Parameter(torch.zeros(parameters_shape))
        
    def forward(self, input):
        dims = [-(i + 1) for i in range(len(self.parameters_shape))]
        mean = input.mean(dim=dims, keepdim=True)
        print(f"Mean \n ({mean.size()}): \n {mean} ")
        
        var = ((input - mean)**2).mean(dim=dims, keepdim=True)
        std = (var + self.eps).sqrt()
        print(f"Standart Deviation \n ({std.size()}): \n {std} ")
        
        y = (input - mean) / std
        
        out = self.gamma * y + self.beta
        print(f"out \n ({out.size()}) = \n {out}")
        
        return out
        

batch_size = 3
sentence_length = 5
embedding_dim = 8 
inputs = torch.randn(sentence_length, batch_size, embedding_dim)

# print(f"input \n ({inputs.size()}) = \n {inputs}")
""" 
# Second simple data for test
inputs = torch.Tensor([[[0.2, 0.1, 0.3], [0.5, 0.1, 0.1]]])
B, S, E = inputs.size()
inputs = inputs.reshape(S, B, E)
"""

layer_norm = LayerNormalization(inputs.size()[-2:])
out = layer_norm.forward(inputs)


        
        
        
        
        
        
        
        