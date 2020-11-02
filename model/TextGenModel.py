import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from collections import Counter
import os
from argparse import Namespace

class LSTMModel(nn.Module):
    def __init__(self, voc_size, seq_size, embed_size, lstm_size):
        super(RNNModule, self).__init__()
        self.seq_size = seq_size
        self.lstm_size = lstm_size
        self.embedding = nn.Embedding(n_vocab, embedding_size)
        self.lstm = nn.LSTM(embedding_size,
                            lstm_size,
                            batch_first=True)
        self.dense = nn.Linear(lstm_size, n_vocab)
    
    def forward(self, x, prev_state):
        output = self.embedding(x)
        output, next_state = self.lstm(embed, prev_state)
        final_output = self.dense(output)
        return final_output, next_state
    
    def zero_state(self, batch_size):
        return (torch.zeros(1, batch_size, self.lstm_size),
                torch.zeros(1, batch_size, self.lstm_size))
    
    def loss_function(net, lr=0.001):
        loss_func = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(net.parametsrs(), lr=lr)
        return criterion, optimizer 
    
    
    