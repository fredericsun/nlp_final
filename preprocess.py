from torch.utils.data import Dataset, DataLoader
import torch
from torch.nn.utils.rnn import pad_sequence
import json


class ModelDataset(Dataset):
    def __init__(self, input_file, tokenizer, max_seq_len, window_stride):
        self.inputs = []
        self.start_pos = []
        self.end_pos = []
        self.token_type = []
        self.context = []
        self.q2con = []

        with open(input_file) as f:
            for data in json.load(f)['data']:
                data = data['paragraphs'][0]
                assert(tokenizer.tokenize(data['context'])[-1] == 'CANNOTANSWER')
                context = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(data['context'])[:-1])
                no_answer = tokenizer.convert_tokens_to_ids(tokenizer.tokenize('CANNOTANSWER'))
                start_offset = 0
                for qas in data['qas']:
                    q = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(qas['question']))

                    context_span_len = max_seq_len - len(q) - len(no_answer) - 3
                    context_span = context[start_offset:start_offset+context_span_len]

                    cur_input = [tokenizer.bos_token_id] + context_span + no_answer + [tokenizer.sep_token_id] + q + [tokenizer.eos_token_id]
                    self.inputs.append(torch.tensor(cur_input))
                    segment_ids = [1] + [0] * (len(context_span) + len(no_answer)) + [1] * (len(q) + 2)
                    self.token_type.append(torch.tensor(segment_ids))

                    answer_start = qas['orig_answer']['answer_start']
                    answer_start = len(tokenizer.tokenize(data['context'][0:answer_start])) - start_offset
                    answer_len = len(tokenizer.tokenize(qas['orig_answer']['text']))
                    answer_end = answer_start + answer_len

                    if 0 <= answer_start < answer_end < len(context_span):
                        self.start_pos.append(torch.tensor(answer_start))
                        self.end_pos.append(torch.tensor(answer_end))
                    else:
                        self.start_pos.append(torch.tensor(len(context_span)))
                        self.end_pos.append(torch.tensor(len(context_span) + len(no_answer)))

                    next_stride = min(window_stride, len(context) - (start_offset + len(context_span)))

                    start_offset += next_stride

                    self.q2con.append(len(self.context))
                self.context.append(data['context'])
        self.inputs = pad_sequence(self.inputs, batch_first=True, padding_value=0)
        self.token_type = pad_sequence(self.token_type, batch_first=True, padding_value=-1)

    def __len__(self):

        return len(self.inputs)

    def __getitem__(self, idx):

        item = {
            "inputs": self.inputs[idx],
            "start_pos": self.start_pos[idx],
            "end_pos": self.end_pos[idx],
            "token_type": self.token_type[idx],
            "q2con": self.q2con[idx]
        }
        return item


def load_dataset(fn, tokenizer, batch_size, max_seq_len, window_stride):
    train_data = ModelDataset(fn[0], tokenizer, max_seq_len, window_stride)
    test_data = ModelDataset(fn[1], tokenizer, max_seq_len, window_stride)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, test_data.context
