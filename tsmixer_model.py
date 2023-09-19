import torch
from functools import reduce

class ResBlock(torch.nn.Module):
  def __init__(self, input_size, hidden_size):
    super().__init__()
    self.t_norm1 = torch.nn.LayerNorm( (input_size[-2], input_size[-1]))
    # self.t_norm1 = torch.nn.BatchNorm1d( input_size[-2])
    self.t_linear1 = torch.nn.Linear( input_size[-2], input_size[-2] )

    self.f_norm1 = torch.nn.LayerNorm( (input_size[-2], input_size[-1]))
    # self.f_norm1 = torch.nn.BatchNorm1d( input_size[-2])
    self.f_linear1 = torch.nn.Linear(input_size[-1], hidden_size)
    self.f_linear2 = torch.nn.Linear(hidden_size, input_size[-1])

  def forward(self, x):
    _x = x
    # Temporal
    x = self.t_norm1(x)
    x = torch.transpose(x, 1, 2)
    x = self.t_linear1(x)
    x = torch.nn.ReLU()(x)
    x = torch.transpose(x, 1, 2)
    x = torch.nn.Dropout()(x)
    res = x + _x

    # Feature
    x = self.f_norm1(res)
    x = self.f_linear1(x)
    x = torch.nn.ReLU()(x)
    x = torch.nn.Dropout()(x)
    x = self.f_linear2(x)
    x = torch.nn.Dropout()(x)
    return x + res

class TSMixer(torch.nn.Module):
  def __init__(self, input_size, hidden_size, output_size, blocks_n):
    super().__init__()
    self.res_blocks = [ ResBlock(input_size, hidden_size) for _ in range(blocks_n) ]
    self.output = torch.nn.Linear(input_size[-2] * input_size[-1], output_size)
    self.input_size = input_size

  def forward(self, x):
    x = reduce(lambda agg, res_block: res_block(agg), self.res_blocks, x)
    # x = torch.transpose(x, 1, 2)
    x = self.output(x.reshape(-1, self.input_size[-2] * self.input_size[-1]))

    return x