import nltk
from nltk.corpus import stopwords
import os

cachedStopWords = stopwords.words("english")

dir = "dataset"
files = os.listdir(dir)
for file in files:
    with open(dir + "/" + file, 'r') as content_file:
        text = content_file.read()
        text = ' '.join([word for word in text.split() if word not in cachedStopWords])
    fileout = open(file, "w")
    fileout.write("dataset/"+text)
    fileout.close

