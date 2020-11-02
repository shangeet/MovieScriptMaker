import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from collections import Counter
import os
from argparse import Namespace
import csv
import pandas as pd
import TextGenModel

dataframe = pd.read_csv("../spongebobQuotesEng.csv", delimiter=' ', quotechar='|', 
                        encoding='utf-8', 
                        names=['EpisodeID', 'EpisodeName', 'Character', 'Quote'])


def stripDescription(string):
    start = string.find("[")
    end = string.find("]")
    while (start != -1):
        string = string.replace(string[start: end+1], "")
        start = string.find("[")
        end = string.find("]")
    return string
         
def get_data_from_file(batch_size, seq_size):
    df = dataframe
    df.dropna(inplace=True)

    quoteList = []
    charList = []
    print("Quote/Char list added...")
    
    curId = None
    
    for index, row in df.iterrows():
        if curId is None:
            curId = row['EpisodeID']
            charList.append("START")
            quoteList.append("")
        elif curId != row['EpisodeID']:
            curId = row['EpisodeID']
            charList.append("END")
            quoteList.append("")
        quote = row['Quote']
        char = row['Character']
        quote = quote.replace("♪", "")
        quote = quote.strip()
        quoteList += quote.split(" ")
        charList.append(char)
    
        
    quote_word_count = Counter(quoteList)
    sorted_vocab_quote = sorted(quote_word_count, 
                          key=quote_word_count.get, reverse=True)
    
    int_to_vocab_quotes = {k: w for k, w in enumerate(sorted_vocab_quote)}
    vocab_to_int_quotes = {w: k for k, w in int_to_vocab_quotes.items()}
    n_vocab_quotes = len(int_to_vocab_quotes)
    
    print("Quote dict...")
    
    char_word_count = Counter(charList)
    sorted_vocab_char = sorted(char_word_count, 
                          key=char_word_count.get, reverse=True)
    int_to_vocab_char = {k: w for k, w in enumerate(sorted_vocab_char)}
    vocab_to_int_char = {w: k for k, w in int_to_vocab_char.items()}
    n_vocab_char = len(int_to_vocab_char)
    
    print("Char dict...")
    
    curId = None
    embedList = []
    for index, row in df.iterrows():
        if curId is None:
            curId = row['EpisodeID']
            embedList.append(["START", ""])
        elif curId != row['EpisodeID']:
            curId = row['EpisodeID']
            embedList.append(["END", ""])
        quote = row['Quote']
        char = row['Character']
        quote = quote.replace("♪", "")
        quote = quote.strip()
        qList = quote.split(" ")
        for word in qList:
            embedList.append([char, word])
    
    print("Embed list finished...")
    
    int_text = [[vocab_to_int_char[w[0]], vocab_to_int_quotes[w[1]]] for w in embedList]

    num_batches = int(len(int_text) / (seq_size * batch_size))
    in_text = int_text[:num_batches * batch_size * seq_size]
    out_text = np.zeros_like(in_text)
    out_text[:-1] = in_text[1:]
    out_text[-1] = in_text[0]
    in_text = np.reshape(in_text, (batch_size, -1))
    out_text = np.reshape(out_text, (batch_size, -1))
    return int_to_vocab, vocab_to_int, n_vocab, in_text, out_text
    

def get_batches(in_text, out_text, batch_size, seq_size):
    num_batches = np.prod(in_text.shape) // (seq_size * batch_size)
    for i in range(0, num_batches * seq_size, seq_size):
        yield in_text[:, i:i+seq_size], out_text[:, i:i+seq_size]

def predict(device, net, words, n_vocab, vocab_to_int, int_to_vocab, top_k=5):
    net.eval()

    state_h, state_c = net.zero_state(1)
    state_h = state_h.to(device)
    state_c = state_c.to(device)
    for w in words:
        ix = torch.tensor([[vocab_to_int[w]]]).to(device)
        output, (state_h, state_c) = net(ix, (state_h, state_c))
    
    _, top_ix = torch.topk(output[0], k=top_k)
    choices = top_ix.tolist()
    choice = np.random.choice(choices[0])

    words.append(int_to_vocab[choice])
            
            
    
    
        
    

if __name__ == '__main__':
    get_data_from_file(64,32)
    # batch_size = 16
    # seq_size = 32
    # lstm_size=64
    # embedding_size=64
    # gradients_norm=5
    # initial_words=['Wonderful', 'day', 'Patrick']
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # int_to_vocab, vocab_to_int, n_vocab, in_text, out_text = get_data_from_file(
    #     batch_size, seq_size)

    # net = TextGenModel(n_vocab, seq_size,
    #                 flags.embedding_size, flags.lstm_size)
    # net = net.to(device)

    # criterion, optimizer = get_loss_and_train_op(net, 0.01)
    

    # iteration = 0
    # for e in range(50):
    #     batches = get_batches(in_text, out_text, batch_size, seq_size)
    #     state_h, state_c = net.zero_state(batch_size)
        
    #     # Transfer data to GPU
    #     state_h = state_h.to(device)
    #     state_c = state_c.to(device)
    #     for x, y in batches:
    #         iteration += 1
            
    #         # Tell it we are in training mode
    #         net.train()

    #         # Reset all gradients
    #         optimizer.zero_grad()

    #         # Transfer data to GPU
    #         x = torch.tensor(x).to(device)
    #         y = torch.tensor(y).to(device)

    #         logits, (state_h, state_c) = net(x, (state_h, state_c))
    #         loss = criterion(logits.transpose(1, 2), y)

    #         state_h = state_h.detach()
    #         state_c = state_c.detach()

    #         loss_value = loss.item()

    #         # Perform back-propagation
    #         loss.backward()

    #         # Update the network's parameters
    #         optimizer.step()
    #         loss.backward()

    #         _ = torch.nn.utils.clip_grad_norm_(
    #             net.parameters(), gradients_norm)

    #         optimizer.step()
    #         if iteration % 100 == 0:
    #             print('Epoch: {}/{}'.format(e, 200),
    #                   'Iteration: {}'.format(iteration),
    #                   'Loss: {}'.format(loss_value))

    #         if iteration % 1000 == 0:
    #             predict(device, net, initial_words, n_vocab,
    #                     vocab_to_int, int_to_vocab, top_k=5)
    #             torch.save(net.state_dict(),
    #                        'checkpoint_pt/model-{}.pth'.format(iteration))
    
    





