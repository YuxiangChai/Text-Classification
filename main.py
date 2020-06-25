import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as data
from models import FastText, CNN, LSTM
from dataset import DataSet, WordDict

import argparse
import time

embed_size = 15
epochs = 5
sentence_len = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, optimizer, criterion, dataloader):
    start = time.time()
    model.to(device)
    model.train()

    epoch_loss = 0

    for i, data in enumerate(dataloader):
        sentence, label = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        pred = model(sentence)
        loss = criterion(pred, label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    print('Loss: ', epoch_loss)
    end = time.time()
    print('Time: ', int(end - start))


def test(model, dataloader):
    model.to(device)
    model.eval()

    correct = 0
    wrong = 0

    for i, data in enumerate(dataloader):
        sentence, label = data[0].to(device), data[1].to(device)
        pred = model(sentence)
        for j in range(len(pred)):
            pred_label = torch.argmax(pred[j])
            if pred_label == label[j]:
                correct += 1
            else:
                wrong += 1

    print('Accuracy: {:.2f}'.format(correct / (correct+wrong)))


def main(args):
    word_dict = WordDict(args.train_file)
    train_set = DataSet(args.train_file, word_dict, sentence_len)
    train_loader = data.DataLoader(train_set, batch_size=128)

    test_set = DataSet(args.test_file, word_dict, sentence_len)
    test_loader = data.DataLoader(test_set, batch_size=128)
    print('finish loading the data')

    if args.model == 'fasttext':
        model = FastText(train_set.vocab_size(), embed_size, 14)
    elif args.model == 'cnn':
        model = CNN(train_set.vocab_size(), embed_size, 14)
    else:
        model = LSTM(train_set.vocab_size(), embed_size, 14)

    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss().to(device)

    for epoch in range(epochs):
        print('Epoch ', epoch)
        train(model, optimizer, criterion, train_loader)
        test(model, test_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, defualt='fasttext')
    parser.add_argument('--train_file', type=str, required=True)
    parser.add_argument('--test_file', type=str, required=True)
    args = parser.parse_args()
    main(args)
