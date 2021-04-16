from torch.utils.data import Dataset, DataLoader
import torch
from torch.nn.utils.rnn import pad_sequence
import json


NUM_HISTORY = 3
CANNOTANSWER = "CANNOTANSWER"


class ModelDataset(Dataset):
    def __init__(self, input_file, tokenizer, max_seq_len, window_stride):
        self.inputs = []
        self.start_pos = []
        self.end_pos = []
        self.token_type = []

        with open(input_file) as f:
            for data in json.load(f)['data']:
                data = data['paragraphs'][0]
                assert(tokenizer.tokenize(data['context'])[-1] == CANNOTANSWER)
                context = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(data['context'])[:-1])
                no_answer = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(CANNOTANSWER))
                for qas in data['qas']:
                    q = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(qas['question']))

                    context_span_len = max_seq_len - len(q) - len(no_answer) - 3

                    start_offset = 0
                    while True:
                        chunk_size = context_span_len
                        if start_offset + chunk_size > len(context):
                            chunk_size = len(context) - start_offset

                        context_span = context[start_offset:start_offset+chunk_size]

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
                        if start_offset + chunk_size >= len(context):
                            break
                        start_offset += window_stride
        self.inputs = pad_sequence(self.inputs, batch_first=True, padding_value=0)
        self.token_type = pad_sequence(self.token_type, batch_first=True, padding_value=1) # TODO

    def __len__(self):

        return len(self.inputs)

    def __getitem__(self, idx):

        item = {
            "inputs": self.inputs[idx],
            "start_pos": self.start_pos[idx],
            "end_pos": self.end_pos[idx],
            "token_type": self.token_type[idx]
        }
        return item


class HAEDataset(Dataset):
    def __init__(self, input_file, tokenizer, max_seq_len, window_stride):
        self.inputs = []
        self.start_pos = []
        self.end_pos = []
        self.token_type = []
        self.history_masks = []

        with open(input_file) as f:
            for data in json.load(f)['data']:
                data = data['paragraphs'][0]
                assert(tokenizer.tokenize(data['context'])[-1] == CANNOTANSWER)
                context = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(data['context'])[:-1])
                no_answer = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(CANNOTANSWER))
                for index, qas in enumerate(data['qas']):
                    ################## Question ##################
                    q = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(qas['question']))

                    ################## History ##################
                    history = data["qas"][max(0, index - NUM_HISTORY): index]
                    history_txt = []
                    for turn in history:
                        history_txt.append(turn["question"])
                        history_txt.append(turn["orig_answer"]["text"])
                    history = tokenizer.encode(" ".join(history_txt))

                    ################## Context Sliding Window ##################
                    context_span_len = max_seq_len - len(q) - len(no_answer) - len(history) - 3

                    start_offset = 0
                    while True:
                        chunk_size = context_span_len
                        if start_offset + chunk_size > len(context):
                            chunk_size = len(context) - start_offset

                        context_span = context[start_offset:start_offset+chunk_size]

                        cur_input = [tokenizer.bos_token_id] + q + [tokenizer.sep_token_id] + history + context_span + no_answer + [tokenizer.eos_token_id]
                        segment_ids = [1] * (len(q) + 2) + [0] * (len(history) + len(context_span) + len(no_answer)) + [1]
                        history_mask = [0] * (len(q) + 2) + [1] * len(history) + [0] * (len(context_span) + len(no_answer)+ 1)

                        self.inputs.append(torch.tensor(cur_input))
                        self.token_type.append(torch.tensor(segment_ids))
                        self.history_masks.append(torch.tensor(history_mask))

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
                        if start_offset + chunk_size >= len(context):
                            break
                        start_offset += window_stride
        self.inputs = pad_sequence(self.inputs, batch_first=True, padding_value=0)
        self.token_type = pad_sequence(self.token_type, batch_first=True, padding_value=1) # TODO
        self.history_masks = pad_sequence(self.history_masks, batch_first=True, padding_value=0)

    def __len__(self):

        return len(self.inputs)

    def __getitem__(self, idx):

        item = {
            "inputs": self.inputs[idx], # Q HISTORY CONTEXT
            "start_pos": self.start_pos[idx],
            "end_pos": self.end_pos[idx],
            "token_type": self.token_type[idx],
            "history_mask": self.history_masks[idx], # Q1[SEP]A1[SEP]Q2[SEP]A2
        }
        return item

def load_dataset(fn, tokenizer, batch_size, max_seq_len, window_stride):
    train_data = ModelDataset(fn[0], tokenizer, max_seq_len, window_stride)
    test_data = ModelDataset(fn[1], tokenizer, max_seq_len, window_stride)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def load_hae_dataset(fn, tokenizer, batch_size, max_seq_len, window_stride):
    train_data = HAEDataset(fn[0], tokenizer, max_seq_len, window_stride)
    test_data = HAEDataset(fn[1], tokenizer, max_seq_len, window_stride)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader