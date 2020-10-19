import requests
from bs4 import BeautifulSoup
import sys
import csv

sys.setrecursionlimit(10000)
URL = "https://www.animecharactersdatabase.com/episodetranscript.php"

page = requests.get(URL, headers={ 'User-Agent' : 'user_agent' })

soup = BeautifulSoup(page.content, 'html.parser')
episodeTable = soup.find(id='besttable')
episodeList = episodeTable.find_all('a')
with open('animeQuotesEng.csv', 'a', newline='', encoding='utf-8') as csvfile:
    quoteWriter = csv.writer(csvfile, delimiter=' ',quotechar='|', quoting=csv.QUOTE_MINIMAL)
    quoteWriter.writerow(['EpisodeID', 'Quote'])
    episodeCount = 1903
    for episode in episodeList:
        episodeCount += 1
        episodeURL = "https://www.animecharactersdatabase.com/" + episode['href']
        episodePage = requests.get(episodeURL, headers={'User-Agent' : 'user_agent'})
        soup = BeautifulSoup(episodePage.content, 'html.parser')
        transcript = soup.find(id='transcript')
        quotes = transcript.find_all('tr')
        for quote in quotes:
            speaker = quote.find('a')
            sentence = quote.find('cite')
            if "&" not in sentence.text:
                quoteWriter.writerow([episodeCount, sentence.text])
