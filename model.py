import torch
import torch.nn as nn
from transformers import GPT2Model

class GPT24QUAC(nn.Module):
    def __init__(self):
        super(GPT24QUAC, self).__init__()
        '''
        Load the pre-trained GPT2 Language Model Head Model
        '''
        self.gpt2 = GPT2Model.from_pretrained("gpt2")
        self.config = self.gpt2.config
        self.head = nn.Linear(self.config.n_embd, 2, bias=True)
        self.loss_func = nn.CrossEntropyLoss()

    """
    input_ids = [batch_size * seq_len]
    token_type_ids = [batch_size * seq_len]
    start_pos = [batch_size * seq_len]
    end_pos = [batch_size * seq_len]

    start_logits = end_logits = [batch_size * seq_len]
    """
    def forward(self, input_ids, token_type_ids=None, start_pos=None, end_pos=None):
        hidden = self.gpt2(input_ids=input_ids, token_type_ids=token_type_ids)[0]
        logits = self.head(hidden)

        start_logits, end_logits = torch.split(logits, 1, dim=2)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        # start_logits = end_logits = [batch_size * seq_len]

        if start_pos is not None and end_pos is not None:
            start_loss = self.loss_func(start_logits, start_pos)
            end_loss = self.loss_func(end_logits, end_pos)
            total_loss = (start_loss + end_loss) / 2
            return total_loss
        else:
            return start_logits, end_logits

    def dummy_inputs(self):
        return self.gpt2.dummy_inputs

    def resize_token_embeddings(self, new_size):
        self.gpt2.resize_token_embeddings(new_size)
