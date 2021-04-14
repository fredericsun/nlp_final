from comet_ml import Experiment
from torch.utils.data import DataLoader
from torch import nn, optim
import torch
import numpy as np
import argparse
from tqdm import tqdm  # optional progress bar
from preprocess import ModelDataset, load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# TODO: Set hyperparameters
hyperparams = {
    "num_epochs": None,
    "batch_size": None,
    "lr": None
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, train_loader, loss_fn, optimizer, experiment, hyperparams):
    """
    Training loop that trains BERT model.

    Inputs:
    - model: A BERT model
    - train_loader: Dataloader of training data
    - experiment: comet.ml experiment object
    - hyperparams: Hyperparameters dictionary
    """
    model.train()
    with experiment.train():
        # TODO: Write training loop
        pass


def test(model, test_loader, loss_fn, word2vec, experiment, hyperparams):
    """
    Testing loop for BERT model and logs perplexity and accuracy to comet.ml.

    Inputs:
    - model: A BERT model
    - train_loader: Dataloader of training data
    - experiment: comet.ml experiment object
    - hyperparams: Hyperparameters dictionary
    """
    model.eval()
    with experiment.test(), torch.no_grad():
        # TODO: Write testing loop
        perplexity = None
        accuracy = None
        experiment.log_metric("perplexity", perplexity)
        experiment.log_metric("accuracy", accuracy)
        pass


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
    # experiment = Experiment(log_code=False)
    # experiment.log_parameters(hyperparams)

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.add_special_tokens({"sep_token": "<SEP>",
                                  "bos_token": "<BOS>",
                                  "eos_token": "<EOS>",
                                  "pad_token": "<PAD>"})

    train_loader, test_loader = load_dataset([args.train_file, args.test_file], tokenizer, batch_size=hyperparams["batch_size"])

    # if args.load:
    #     model.load_state_dict(torch.load('./model.pt'))
    # if args.train:
    #     train(model, train_loader, loss_fn, optimizer, word2vec, experiment,
    #           hyperparams)
    # if args.test:
    #     test(model, test_loader, loss_fn, word2vec, experiment, hyperparams)
    # if args.save:
    #     torch.save(model.state_dict(), './model.pt')
