from pysubparser import parser
from pysubparser.cleaners import ascii, brackets, formatting
import glob
import csv

def cleanSubtitles(subtitles):
    return brackets.clean(formatting.clean(subtitles))

baseFilePath = "assEng/*.ass"
fileList = glob.glob(baseFilePath)
episodeCount = 0
with open('animeQuotesEng.csv', 'a', newline='', encoding='utf-8') as csvfile:
    quoteWriter = csv.writer(csvfile, delimiter=' ',quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for path in fileList:
        episodeCount += 1
        filepath = "./" + path
        subtitles = parser.parse(filepath)
        subtitles = cleanSubtitles(subtitles)
        for sub in subtitles:
            if "{" not in sub.text and sub.text.strip() != "":
                quoteWriter.writerow([episodeCount, sub.text.strip()])
                #print(sub.text, episodeCount)
