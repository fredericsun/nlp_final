import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel


class GPT2(nn.Module):
    def __init__(self, *args):
        super(GPT2, self).__init__()
        '''
        Load the pre-trained GPT2 Language Model Head Model
        '''
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')

    def forward(self, input, label):
        outputs = self.gpt(input_ids=input, labels=label)
        return outputs[0]