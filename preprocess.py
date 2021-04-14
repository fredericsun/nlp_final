from torch.utils.data import Dataset, DataLoader
import torch
from torch.nn.utils.rnn import pad_sequence


class ModelDataset(Dataset):
    def __init__(self, input_file, tokenizer, gpt=False):
        self.inputs = []
        self.labels = []
        self.length = []

        data = []
        with open(input_file) as f:
            for line in f.readlines():
                data.append(line.strip())
        data = data[1:]

        for line in data:
            cur_label = tokenizer.encode(line)
            if not gpt:
                line = "START " + line[:-5]
            cur_input = tokenizer.encode(line)
            self.length.append(len(cur_input))
            self.inputs.append(torch.LongTensor(cur_input))
            self.labels.append(torch.LongTensor(cur_label))

        self.inputs = pad_sequence(self.inputs, batch_first=True)
        if gpt:
            self.labels = pad_sequence(
                self.labels, batch_first=True, padding_value=-100)
        else:
            self.labels = pad_sequence(self.labels, batch_first=True)

    def __len__(self):

        return len(self.inputs)

    def __getitem__(self, idx):

        item = {
            "inputs": self.inputs[idx],
            "labels": self.labels[idx],
            "length": self.length[idx]
        }
        return item


def load_dataset(fn, tokenizer, batch_size, gpt=False):
    train_data = ModelDataset(fn[0], tokenizer, gpt)
    test_data = ModelDataset(fn[1], tokenizer, gpt)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
