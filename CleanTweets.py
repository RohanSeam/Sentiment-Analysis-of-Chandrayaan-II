# Rohan Seam
# 8/23/2019
# Final Project
# The purpose of this script is to clean tweets that are saved in a text file.

import re 
import string
from nltk.corpus import stopwords

def clean_doc(doc):
	# split into tokens by white space
	tokens = doc.split()
	# prepare regex for char filtering
	re_punc = re.compile('[%s]' % re.escape(string.punctuation))
	# remove punctuation from each word
	tokens = [re_punc.sub('' , w) for w in tokens]
	# remove remaining tokens that are not alphabetic
	tokens = [word for word in tokens if word.isalpha()]
	# filter out stop words
	stop_words = set(stopwords.words('english'))
	tokens = [w for w in tokens if not w in stop_words]
	# filter out short tokens
	tokens = [word for word in tokens if len(word) > 1]
	return ' '.join(tokens)

inputFile = open('NEWEST TWEETS ROHAN/tweets.txt', 'r', encoding="utf8")
outputFile = open('NEWEST TWEETS ROHAN/clean_tweets.txt', 'w', encoding="utf8")

for tweet in inputFile:
    cleaned = clean_doc(tweet)
    result = re.sub(r"http\S+", "", cleaned)
    result = re.sub(r"@\S+", "", cleaned)
    if result:
        outputFile.write("\"" + str(result) + "\"," + '\n')
    
inputFile.close()
outputFile.close()