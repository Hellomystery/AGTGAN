# encoding: utf-8

import torch.nn.functional as F
from torch.autograd import Variable

def grid_sample(input, grid, canvas = None, padding_mode='zeros'):
    output = F.grid_sample(input, grid, padding_mode=padding_mode)
    if canvas is None:
        return output
    else:
        input_mask = Variable(input.data.new(input.size()).fill_(1))
        output_mask = F.grid_sample(input_mask, grid)
        padded_output = output * output_mask + canvas * (1 - output_mask)
        return padded_output
