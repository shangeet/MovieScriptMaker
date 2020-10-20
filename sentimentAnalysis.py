# -*- coding: utf-8 -*-
import flair
import csv

def getCharacter(sentimentValue):
    decScore = sentimentValue.score
    decTemper = sentimentValue.value
    if decTemper == "POSITIVE":
        if decScore >= 0.75:
            return 'D'
        return 'C'
    else:
        if decScore >= 0.75:
            return 'A'
        return 'B'

flair_sentiment = flair.models.TextClassifier.load('en-sentiment')  

with open("animeQuotesEng.csv", newline='', encoding='utf-8') as csvfile:
    with open('animeQuotesEngChar.csv', 'a', newline='', encoding='utf-8') as newcsvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar="|")
        writer = csv.writer(newcsvfile, delimiter=' ',quotechar='|', quoting=csv.QUOTE_MINIMAL)
        idx = 0
        for row in reader:
            if idx > 635495 and len(row) > 1:
                s = flair.data.Sentence(row[1])
                if len(s) > 0:
                    flair_sentiment.predict(s)
                    total_sentiment = s.labels
                    character = getCharacter(total_sentiment[0])
                    newRow = row + [character]
                    writer.writerow(newRow)
            idx += 1
    