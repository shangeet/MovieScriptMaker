# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 17:15:06 2020

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
        sentence = model.make_sentence(init_state=previousState, max_words=10)
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
                sentence = model.make_sentence(init_state=previousState, max_words=10)
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
                
        sentence = model.make_sentence(init_state=None, max_words=10)
        previousSpeaker = speaker
        words = sentence.split(" ")
        previousState = tuple([words[second], words[first]])
        return sentence, previousSpeaker, previousState, previousSentence

def cleanName(name):
    if '.' in name:
        name = name.split(".")[0]
    return name
    
def getCharacterMappings(script):
    mappings = ["A", "B", "C", "D"]
    charMappings = {}
    for pair in script:
        if len(charMappings.keys()) == 4:
            return charMappings
        speaker, sentence = pair
        wordList = nltk.word_tokenize(sentence)
        cleanedTokenizedList = []
        for w in wordList:
            lw = w.lower()
            if lw in words.words():
                cleanedTokenizedList.append(lw)
            else:
                cleanedTokenizedList.append(w)
            
        tags = nltk.pos_tag(cleanedTokenizedList)
        for tag in tags:
            if(tag[1] == "NNP"):
                possibleMappings = set(mappings)
                possibleMappings.remove(pair[0])
                for key in charMappings.keys():
                    if key in possibleMappings:
                        possibleMappings.remove(key)
                if(len(possibleMappings) > 0 and tag[0] not in charMappings.values() and "…" not in tag[0] and "’" not in tag[0] and "-" not in tag[0]):
                    charMappings[possibleMappings.pop()] = cleanName(tag[0])
                    
    for mapping in mappings:
        if mapping not in charMappings.keys():
            charMappings[mapping] = mapping
    return charMappings
        
    
def main(filepath, numLines):
    markovChain = MarkovChain(filepath)
    print("Building model for characters...")
    modelA = markovChain.buildModelForPesonality("A")
    modelB = markovChain.buildModelForPesonality("B")
    modelC = markovChain.buildModelForPesonality("C")
    modelD = markovChain.buildModelForPesonality("D")
    print("Characters models finished...")
    print("Generating script...")
    print("||||||||||||||||||||||||||||||||")
    
    scenes = ["*At a cafe*", "*In senpai's room*", "*At school*"]
    characters = ["A", "B", "C", "D"]
    charModelMap = {"A" : modelA, "B" : modelB, "C" : modelC, "D" : modelD}
    
    print(random.choice(scenes))
    previousSpeaker = None
    previousState = None
    previousSentence = None
    
    script = []
    
    for i in range(int(numLines)):
        speaker = selectCharacter(characters, previousSpeaker)
        model = charModelMap[speaker]
        quote, previousSpeaker, previousState, previousSentence = generateSentence(model, speaker, previousSpeaker, previousState, previousSentence)
        script.append((speaker, quote))
        
    charMappings = getCharacterMappings(script)
    for vals in script:
        speaker = charMappings[vals[0]]
        quote = vals[1]
        print(speaker + ":", quote)
        print("")
    

    print("||||||||||||||||||||||||||||||||")

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])