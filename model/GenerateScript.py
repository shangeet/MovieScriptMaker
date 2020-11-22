# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 17:16:05 2020

@author: Shantanu
"""
import torch
import torch.nn.functional as F
import numpy as np
import pickle

sequence_length = 10  # of words in a sequence
train_on_gpu = True



def generate(rnn, prime_id, int_to_vocab, token_dict, pad_value, predict_len, vocab_to_int):
    """
    Generate text using the neural network
    :param decoder: The PyTorch Module that holds the trained neural network
    :param prime_id: The word id to start the first prediction
    :param int_to_vocab: Dict of word id keys to word values
    :param token_dict: Dict of puncuation tokens keys to puncuation values
    :param pad_value: The value used to pad a sequence
    :param predict_len: The length of text to generate
    :return: The generated text
    """
    rnn.eval()
    # create a sequence (batch_size=1) with the prime_id
    current_seq = np.full((1, sequence_length), pad_value)
    current_seq[-1][-1] = prime_id
    predicted = [int_to_vocab[prime_id]]
    
    endMarkerFound = False
    
    while not endMarkerFound:
        if train_on_gpu:
            current_seq = torch.LongTensor(current_seq).cuda()
        else:
            current_seq = torch.LongTensor(current_seq)
        
        # initialize the hidden state
        hidden = rnn.init_hidden(current_seq.size(0))
        
        # get the output of the rnn
        output, _ = rnn(current_seq, hidden)
        
        # get the next word probabilities
        p = F.softmax(output, dim=1).data
        if(train_on_gpu):
            p = p.cpu() # move to cpu
         
        # use top_k sampling to get the index of the next word
        top_k = 5
        p, top_i = p.topk(top_k)
        top_i = top_i.numpy().squeeze()
        
        # select the likely next word index with some element of randomness
        p = p.numpy().squeeze()
        word_i = np.random.choice(top_i, p=p/p.sum())
        
        # retrieve that word from the dictionary
        word = int_to_vocab[word_i]
        predicted.append(word)
        
        # the generated word becomes the next "current sequence" and the cycle can continue
        current_seq = np.roll(current_seq.cpu(), -1, 1)
        current_seq[-1][-1] = word_i
        
        for word in current_seq[0]:
            if vocab_to_int['<END>'] == word:
                endMarkerFound = True
    
    gen_sentences = ' '.join(predicted)
    # Replace punctuation tokens
    for key, token in token_dict.items():
        ending = ' ' if key in ['\n', '(', '"'] else ''
        gen_sentences = gen_sentences.replace(' ' + token.lower(), key)
    gen_sentences = gen_sentences.replace('\n ', '\n')
    gen_sentences = gen_sentences.replace('( ', '(')
    
    for punc, val in token_dict.items():
        gen_sentences = gen_sentences.replace(val, punc)
    
    # return all the sentences
    return gen_sentences


def generate_script(sentences):
    sentenceList = []
    
    wordList = sentences.split(" ")
    sentence = ""
    for word in wordList:
        if word == "<SQ>":
            sentence = ""
        elif word == "<EQ>":
            sentenceList.append(sentence)
        else:
            sentence += word + " "
            
    for sent in sentenceList:
        sent = ' '.join(sent.split())
        print(sent)
    
if __name__ == '__main__':

    int_text, vocab_to_int, int_to_vocab, punt_dict = pickle.load(open('preprocess3.p', mode='rb'))
    print(len(int_text))
    print("====================")
    print(len(vocab_to_int))
    print("====================")
    print(len(int_to_vocab))
    print("====================")
    print("Loaded vocab dicts...")
    rnn = torch.load("trained_rnn_3.pt")
    print("Loaded model...")
    rnn.cuda()
    print("Running on CUDA...")
    prime_id = vocab_to_int['<SQ>']
    prime_word = "<SQ>"
    padValue = len(vocab_to_int)
    genLength = 200
    sentences = generate(rnn, prime_id, int_to_vocab, punt_dict, padValue, genLength, vocab_to_int)
    generate_script(sentences)
    
    
    
    
    
    
    
    
    
    
    