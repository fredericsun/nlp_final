from comet_ml import Experiment
from torch.utils.data import DataLoader
from torch import nn, optim
import torch
import numpy as np
import argparse
from tqdm import tqdm  # optional progress bar
from preprocess import load_hae_dataset
from transformers import GPT2Tokenizer, BertTokenizer
from bertModel import BERT4QUAC
from scorer import f1_score

# TODO: Set hyperparameters
hyperparams = {
    "num_epochs": 10,
    "batch_size": 10,
    "lr": 0.001,
    "max_seq_len": 512,
    "window_stride": 64
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
                history_mask = batch["history_mask"].to(device)
                loss = model(inputs, token_type_ids=token_type, start_positions=start_pos, end_positions=end_pos, history_type_ids=history_mask)[0]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                experiment.log_metric("loss", loss.cpu().detach().numpy())


def test(model, test_loader, tokenizer, experiment, hyperparams):
    model.eval()
    with experiment.test(), torch.no_grad():
        f1_sum = 0
        f1_count = 0
        for batch in tqdm(test_loader):
            inputs = batch['inputs'].to(device)
            start_pos = batch['start_pos'].to(device)
            end_pos = batch['end_pos'].to(device)
            token_ids = batch['token_type'].to(device)
            history_mask = batch["history_mask"].to(device)
            context = []
            for token_id, input in zip(token_ids, inputs):
                context_id = []
                for index, id in enumerate(token_id):
                    if id == 0:
                        context_id.append(input[index])
                context.append(context_id)

            start_logits, end_logits = model(inputs, token_type_ids=token_ids, history_type_ids=history_mask)[:2]
            pred_start = np.argmax(start_logits.cpu().detach().numpy(), axis=1)
            pred_end = np.argmax(end_logits.cpu().detach().numpy(), axis=1)
            ground_truth = id_to_text(tokenizer, context, start_pos.cpu().detach().numpy(), end_pos.cpu().detach().numpy())
            prediction = id_to_text(tokenizer, context, pred_start, pred_end)
            for pred, truth in zip(prediction, ground_truth):
                f1_sum += f1_score(pred, truth)
                f1_count += 1
        f1_avg = f1_sum / f1_count
        experiment.log_metric("f1", f1_avg)


def id_to_text(tokenizer, context, start_pos, end_pos):
    result = []
    for index, (i, j) in enumerate(zip(start_pos, end_pos)):
        if i < 0 or j >= len(context[index]) or i >= j:
            result.append('CANNOTANSWER')
        else:
            result.append(tokenizer.decode(context[index][i:j+1]))
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

    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    tokenizer.add_special_tokens({"sep_token": "<SEP>",
                                    "bos_token": "<BOS>",
                                    "eos_token": "<EOS>",
                                    "pad_token": "<PAD>"})
    tokenizer.add_tokens("CANNOTANSWER")
    model = BERT4QUAC(len(tokenizer)).to(device)

    train_loader, test_loader = load_hae_dataset([args.train_file, args.test_file], tokenizer,
                                            batch_size=hyperparams["batch_size"],
                                            max_seq_len=hyperparams['max_seq_len'],
                                            window_stride=hyperparams['window_stride'])

    optimizer = optim.Adam(model.parameters(), lr=hyperparams['lr'])

    if args.load:
        model.load_state_dict(torch.load('./model.pt'))
    if args.train:
        train(model, train_loader, optimizer, experiment, hyperparams)
    if args.test:
        test(model, test_loader, tokenizer, experiment, hyperparams)
    if args.save:
        torch.save(model.state_dict(), './model.pt')
