# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 16:45:27 2020

@author: Shantanu
"""

import pandas as pd
import markovify
import en_core_web_sm

nlp = en_core_web_sm.load()

class MarkovChain(markovify.Text):
    
    def __init__(self, filepath):
        self.filepath = filepath
        self.dataframe = pd.read_csv(filepath, delimiter=' ', quotechar='|', encoding='utf-8', names=['EpisodeID', 'EpisodeName', 'Character', 'Quote'])
        
    def buildModelForPesonality(self, personalityType):
        dfPersonality = self.dataframe.loc[self.dataframe['Personality'] == personalityType]
        totalText = ""
        for quote in dfPersonality['Quote']:
            quote = quote.strip()
            quote = quote.replace("}", " ")
            quote = quote.replace("{", " ")
            totalText += quote
        model = markovify.Text(totalText)
        return model
    
    def buildModelForCharacter(self, characterName):
        dfCharacter = self.dataframe.loc[self.dataframe['Character'].str.match(characterName)]
        totalText = ""
        model = None
        for quote in dfCharacter['Quote']:
            try:
                quote = quote.replace("♪", "")
                quote = quote.strip()
                if "[" not in quote and "]" not in quote and quote != "":
                    totalText += quote
            except:
                pass
        model = markovify.Text(totalText)
        return model
    
    def word_split(self, sentence):
        return ["::".join((word.orth_, word.pos_)) for word in nlp(sentence)]

    def word_join(self, words):
        sentence = " ".join(word.split("::")[0] for word in words)
        return sentence
        