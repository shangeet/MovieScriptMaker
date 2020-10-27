# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 13:30:40 2020

@author: Shantanu
"""

import sys
from MarkovChain import MarkovChain
import random
import nltk
from nltk.corpus import words

nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('words')

def selectCharacter(characters, previousCharacter):
    while True:
        newCharacter = random.choice(characters)
        if newCharacter != previousCharacter:
            return newCharacter
        
def generateSentence(model, speaker, previousSpeaker, previousState, previousSentence):
    if previousState is None:
        sentence = model.make_sentence(init_state=previousState, max_words=20)
        previousSpeaker = speaker
        words = sentence.split(" ")
        previousState = tuple([words[-2], words[-1]])
        previousSentence = sentence
        return sentence, previousSpeaker, previousState, previousSentence
    else:
        first = -1
        second = -2
        senLen = len(previousSentence.split(" "))
        while(abs(second) < senLen):
            try:
                sentence = model.make_sentence(init_state=previousState, max_words=20)
                previousSpeaker = speaker
                words = sentence.split(" ")
                senLen = len(words)
                previousState = tuple([words[second], words[first]])
                previousSentence = sentence
                return sentence, previousSpeaker, previousState, previousSentence
            except:
                first -= 1
                second -= 1
                words = previousSentence.split(" ")
                previousState = tuple([words[second], words[first]])
                
        sentence = model.make_sentence(init_state=None, max_words=20)
        previousSpeaker = speaker
        try:
            words = sentence.split(" ")
            previousState = tuple([words[second], words[first]])
        except:
            previousState = None
        if sentence is None:
            sentence = model.make_sentence()
        return sentence, previousSpeaker, previousState, previousSentence
    
def main(filepath, numLines):
    markovChain = MarkovChain(filepath)
    print("Building model for characters...")
    modelA = markovChain.buildModelForCharacter("SpongeBob")
    modelB = markovChain.buildModelForCharacter("Patrick")
    modelC = markovChain.buildModelForCharacter("Sandy")
    modelD = markovChain.buildModelForCharacter("SCENE")
    print("Characters models finished...")
    print("Generating script...")
    print("||||||||||||||||||||||||||||||||")
    
    characters = ["SpongeBob", "Patrick", "Sandy", "SCENE"]
    charModelMap = {"SpongeBob" : modelA, "Patrick" : modelB, "Sandy" : modelC, "SCENE" : modelD}
    
    previousSpeaker = None
    previousState = None
    previousSentence = None
    
    script = []
    
    for i in range(int(numLines)):
        speaker = selectCharacter(characters, previousSpeaker)
        model = charModelMap[speaker]
        quote, previousSpeaker, previousState, previousSentence = generateSentence(model, speaker, previousSpeaker, previousState, previousSentence)
        
        script.append((speaker, quote))
        
    for vals in script:
        speaker = vals[0]
        quote = vals[1]
        if speaker == "SCENE":
            print(quote)
            print("")
        else:
            print(speaker + ":", quote)
            print("")
    

    print("||||||||||||||||||||||||||||||||")

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])