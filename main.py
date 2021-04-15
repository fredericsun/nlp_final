from comet_ml import Experiment
from torch.utils.data import DataLoader
from torch import nn, optim
import torch
import numpy as np
import argparse
from tqdm import tqdm  # optional progress bar
from preprocess import load_dataset
from transformers import GPT2Tokenizer, GPT2Config
from model import GPT24QUAC
from scorer import f1_score

# TODO: Set hyperparameters
hyperparams = {
    "num_epochs": 10,
    "batch_size": 100,
    "lr": 0.001,
    "max_seq_len": 1024,
    "window_stride": 128
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, train_loader, optimizer, experiment, hyperparams):
    model.train()
    with experiment.train():
        epoch = hyperparams['num_epochs']
        for i in range(epoch):
            for batch in tqdm(train_loader):
                inputs = batch['inputs'].to(device)
                start_pos = batch['start_pos'].to(device)
                end_pos = batch['end_pos'].to(device)
                token_type = batch['token_type'].to(device)
                loss = model(inputs, token_type_ids=token_type, start_pos=start_pos, end_pos=end_pos)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                experiment.log_metric("loss", loss.cpu().detach().numpy())


def test(model, test_loader, test_context, experiment, hyperparams):
    model.eval()
    with experiment.test(), torch.no_grad():
        f1_sum = 0
        f1_count = 0
        for batch in tqdm(test_loader):
            inputs = batch['inputs'].to(device)
            start_pos = batch['start_pos'].to(device)
            end_pos = batch['end_pos'].to(device)
            q2con = batch['q2con'].to(device)
            context = [test_context[q2con[i]] for i in q2con]
            start_logits, end_logits = model(inputs)
            pred_start = np.argmax(start_logits.cpu().detach().numpy(), axis=1)
            pred_end = np.argmax(end_logits.cpu().detach().numpy(), axis=1)
            ground_truth = id_to_text(context, start_pos.cpu().detach().numpy(), end_pos.cpu().detach().numpy())
            prediction = id_to_text(context, pred_start, pred_end)
            f1_sum += f1_score(prediction, ground_truth)
            f1_count += 1
        f1_avg = f1_sum / f1_count
        experiment.log_metric("f1", f1_avg)


def id_to_text(context, start_pos, end_pos):
    result = []
    for index, (i, j) in enumerate(zip(start_pos, end_pos)):
        if i < 0 or j >= len(context) or i >= j:
            result.append('CANNOTANSWER')
        else:
            result.append(context[index][i:j+1])
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_file")
    parser.add_argument("test_file")
    parser.add_argument("-l", "--load", action="store_true",
                        help="load model.pt")
    parser.add_argument("-s", "--save", action="store_true",
                        help="save model.pt")
    parser.add_argument("-T", "--train", action="store_true",
                        help="run training loop")
    parser.add_argument("-t", "--test", action="store_true",
                        help="run testing loop")
    args = parser.parse_args()

    # TODO: Make sure you modify the `.comet.config` file
    experiment = Experiment(log_code=False)
    experiment.log_parameters(hyperparams)

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    tokenizer.add_special_tokens({"sep_token": "<SEP>",
                                  "bos_token": "<BOS>",
                                  "eos_token": "<EOS>",
                                  "pad_token": "<PAD>"})
    tokenizer.add_tokens("CANNOTANSWER")

    configuration = GPT2Config()

    model = GPT24QUAC()
    model.resize_token_embeddings(len(tokenizer) + 1)

    train_loader, test_loader, test_context = load_dataset([args.train_file, args.test_file], tokenizer,
                                                           batch_size=hyperparams["batch_size"],
                                                           max_seq_len=hyperparams['hyperparams'],
                                                           window_stride=hyperparams['window_stride'])

    optimizer = optim.Adam(model.parameters(), lr=hyperparams['lr'])

    if args.load:
        model.load_state_dict(torch.load('./model.pt'))
    if args.train:
        train(model, train_loader, optimizer, experiment, hyperparams)
    if args.test:
        test(model, test_loader, test_context, experiment, hyperparams)
    if args.save:
        torch.save(model.state_dict(), './model.pt')
