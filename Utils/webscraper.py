import requests
from bs4 import BeautifulSoup
import sys
import csv

sys.setrecursionlimit(10000)
URL = "https://spongebob.fandom.com/wiki/List_of_transcripts"

page = requests.get(URL, headers={ 'User-Agent' : 'user_agent' })

soup = BeautifulSoup(page.content, 'html.parser')
mainDiv = soup.find(id='mw-content-text')
tables = mainDiv.findAll("table", {"class" : "wikitable"})[:13]
with open('spongebobQuotesEng.csv', 'a', newline='', encoding='utf-8') as csvfile:
    quoteWriter = csv.writer(csvfile, delimiter=' ',quotechar='|', quoting=csv.QUOTE_MINIMAL)
    quoteWriter.writerow(['EpisodeID', 'EpisodeName', 'Character', 'Quote'])
    for table in tables:
        rows = table.find_all("tr")
        rows = rows[1:]
        for row in rows:
            info = row.find_all("td")
            episodeID = info[0].text.strip()
            episodeName = info[1].text.strip()
            transcriptURL = "https://spongebob.fandom.com" + info[2].find('a')['href']
            transcript = requests.get(transcriptURL, headers={'User-Agent' : 'user_agent'})
            soup = BeautifulSoup(transcript.content, 'html.parser')
            mainDiv = soup.find(id='mw-content-text')
            quotes = mainDiv.findAll('ul')
            for lines in quotes:
                for line in lines:
                    lineContent = line.text.split(":")
                    if len(lineContent) == 2:
                        characterName = lineContent[0].strip()
                        characterQuote = lineContent[1].strip()
                        quoteWriter.writerow([episodeID, episodeName.strip(), characterName, characterQuote])
                    elif len(lineContent) == 1:
                        characterName = "SCENE"
                        characterQuote = lineContent[0].strip()
                        quoteWriter.writerow([episodeID, episodeName.strip(), characterName, characterQuote])
            print("Finished writing episode: ", episodeID)
    


#episodeList = episodeTable.find_all('a')
# =============================================================================
# with open('spongebobQuotesEng.csv', 'a', newline='', encoding='utf-8') as csvfile:
#     quoteWriter = csv.writer(csvfile, delimiter=' ',quotechar='|', quoting=csv.QUOTE_MINIMAL)
#     quoteWriter.writerow(['EpisodeID', 'EpisodeName', 'Character', 'Quote'])
#     episodeCount = 1903
#     for episode in episodeList:
#         episodeCount += 1
#         episodeURL = "https://spongebob.fandom.com" + episode['href']
#         episodePage = requests.get(episodeURL, headers={'User-Agent' : 'user_agent'})
#         soup = BeautifulSoup(episodePage.content, 'html.parser')
#         transcript = soup.find(id='transcript')
#         quotes = transcript.find_all('tr')
#         for quote in quotes:
#             speaker = quote.find('a')
#             sentence = quote.find('cite')
#             if "&" not in sentence.text:
#                 quoteWriter.writerow([episodeCount, sentence.text])
# =============================================================================
