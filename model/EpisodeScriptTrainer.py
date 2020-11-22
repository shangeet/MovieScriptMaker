import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import pickle
import numpy as np
from collections import Counter
import os
from argparse import Namespace
import csv
import pandas as pd
from model.TextGenModel import RNN
import gc

dataframe = pd.read_csv("../spongebobQuotesEng.csv", delimiter=' ', quotechar='|', 
                        encoding='utf-8', 
                        names=['EpisodeID', 'EpisodeName', 'Character', 'Quote'])

punt_dict = {
        '.': '<PERIOD>',
        ',': '<COMMA>',
        '"': '<QUOTATION_MARK>',
        ';': '<SEMICOLON>',
        '!': '<EXCLAMATION_MARK>',
        '?': '<QUESTION_MARK>',
        '(': '<LEFT_PAREN>',
        ')': '<RIGHT_PAREN>',
        '--': '<HYPHENS>',
        '?': '<QUESTION_MARK>',
        '\n': '<NEW_LINE>',
        ':': '<COLON>',
        '-': '<DASH>',
        '[': '<RIGHT_BRAC>',
        ']': '<LEFT_BRAC>'
    }


def preprocess(quote):
    quote = quote.replace("♪", "")
    for key, token in punt_dict.items():
        quote = quote.replace(key, ' {} '.format(token))
    quote = ' '.join(quote.split())
    return quote
    
def get_data_from_file(batch_size, seq_size):
    df = dataframe
    df.dropna(inplace=True)

    quoteList = []
    
    curId = None
    episodeCount = 0
    
    for index, row in df.iterrows():
        if episodeCount <= 175:
            if curId is None:
                curId = row['EpisodeID']
                quoteList.append("<START>")
            elif curId != row['EpisodeID']:
                curId = row['EpisodeID']
                quoteList.append("<END>")
                episodeCount += 1
            quote = preprocess(row['Character'] + ":" + row['Quote'])
            qList = quote.split(" ")
            qListFinal = ["<SQ>"] + qList + ["<EQ>"]
            quoteList += qListFinal
    quoteList.append('<PAD>')
    print(curId)
    word_count = Counter(quoteList)
    sorted_vocab = sorted(word_count, 
                          key=word_count.get, reverse=True)
    
    int_to_vocab = {k: w for k, w in enumerate(sorted_vocab)}
    vocab_to_int = {w: k for k, w in int_to_vocab.items()}
    n_vocab = len(int_to_vocab)

    int_text = list([vocab_to_int[w] for w in quoteList])
    return int_to_vocab, vocab_to_int, n_vocab, int_text

def get_target_feature(text,x, sequence_length):
    feature = text[x:x+sequence_length]
    target = text[x+sequence_length]
    return feature,target

def make_features_targets(text, sequence_length):
    features = []
    targets = []
    for i in range(len(text)):
        if i+sequence_length<len(text):
            feature,target = get_target_feature(text,i,sequence_length)
            features.append(feature)
            targets.append(target)
    features=np.array(features)
    targets=np.array(targets)
    return features,targets

def batch_data(words, sequence_length, batch_size):
    """
    Batch the neural network data using DataLoader
    :param words: The word ids of the TV scripts
    :param sequence_length: The sequence length of each batch
    :param batch_size: The size of each batch; the number of sequences in a batch
    :return: DataLoader with batched data
    """
    # TODO: Implement function
    features,targets = make_features_targets(words, sequence_length)
    data = TensorDataset(torch.from_numpy(features),torch.from_numpy(targets))
    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size)
    # return a dataloader
    return data_loader    

def forward_back_prop(rnn, optimizer, criterion, inp, target, hidden):
    """
    Forward and backward propagation on the neural network
    :param decoder: The PyTorch Module that holds the neural network
    :param decoder_optimizer: The PyTorch optimizer for the neural network
    :param criterion: The PyTorch loss function
    :param inp: A batch of input to the neural network
    :param target: The target output for the batch of input
    :return: The loss and the latest hidden state Tensor
    """
    
    # TODO: Implement Function
    if(train_on_gpu):
        inp, target = inp.cuda(), target.cuda()
        rnn.cuda()
    h = tuple([each.data for each in hidden])
    rnn.zero_grad()
    output, h = rnn(inp, h)
    loss = criterion(output, target)
    loss.backward()
    nn.utils.clip_grad_norm(rnn.parameters(), 5)
    optimizer.step()
    
    return loss.item(), h

def train_rnn(rnn, batch_size, optimizer, criterion, n_epochs, show_every_n_batches=10):
    batch_losses = []
    
    rnn.train()

    print("Training for %d epoch(s)..." % n_epochs)
    for epoch_i in range(1, n_epochs + 1):
        
        # initialize hidden state
        hidden = rnn.init_hidden(batch_size)
        
        for batch_i, (inputs, labels) in enumerate(train_loader, 1):
            
            # make sure you iterate over completely full batches, only
            n_batches = len(train_loader.dataset)//batch_size
            if(batch_i > n_batches):
                break
            labels = labels.clone().to(torch.int64)
            inputs = inputs.clone().to(torch.int64)
            
            # forward, back prop
            loss, hidden = forward_back_prop(rnn, optimizer, criterion, inputs, labels, hidden)          
            # record loss
            batch_losses.append(loss)

            # printing loss stats
            if batch_i % show_every_n_batches == 0:
                print('Epoch: {:>4}/{:<4}  Loss: {}\n'.format(
                    epoch_i, n_epochs, np.average(batch_losses)))
                batch_losses = []

    # returns a trained rnn
    return rnn
            
def save_model(filename, decoder):
    save_filename = os.path.splitext(os.path.basename(filename))[0] + '.pt'
    torch.save(decoder, save_filename)
    
if __name__ == '__main__':
    batch_size = 128
    sequence_length = 10  # of words in a sequence

    train_on_gpu = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    int_to_vocab, vocab_to_int, n_vocab, int_text = get_data_from_file(
        batch_size, sequence_length)
    
    pickle.dump((int_text, vocab_to_int, int_to_vocab, punt_dict), open('preprocess3.p', 'wb'))

    train_loader = batch_data(int_text, sequence_length, batch_size)
      
    # Training parameters
    
    # Number of Epochs
    num_epochs = 20
    
    # Learning Rate
    learning_rate = 0.001
    
    # Model parameters
    # Vocab size
    vocab_size = len(vocab_to_int) + 1
    print("Vocab Size: ", vocab_size)
    
    # Output size
    output_size = vocab_size + 1
    print("Output Size: ", output_size)
    
    # Embedding Dimension
    embedding_dim = 256
    
    # Hidden Dimension
    hidden_dim = 512
    
    # Number of RNN Layers
    n_layers = 2
    
    # Show stats for every n number of batches
    show_every_n_batches = 10
    
    # create model and move to gpu if available
    rnn = RNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout=0.5)
    if torch.cuda.is_available():
        rnn.cuda()
    else:
        print("Training on CPU...")
    
    # defining loss and optimization functions for training
    optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # training the model
    trained_rnn = train_rnn(rnn, batch_size, optimizer, criterion, num_epochs, show_every_n_batches)
    
    # saving the trained model
    save_model('trained_rnn_3', trained_rnn)
    print('Model Trained and Saved')
    
    sameModel = torch.load('trained_rnn_3.pt')
    print(sameModel)

    
    
    





