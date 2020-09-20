# Rohan Seam
# 8/23/2019
# Final Project
# The purpose of this script is to separate clean tweets into positive and negative text files based 
# on their polarity score.

from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()

with open('NEWEST TWEETS ROHAN/clean_tweets.txt', 'r', encoding="utf8") as f:
    comments = f.readlines()
pos_comments = [comment for comment in comments
                if sid.polarity_scores(comment)['compound'] > 0.5]
neg_comments = [comment for comment in comments
                if sid.polarity_scores(comment)['compound'] < -0.1]
neu_comments = [comment for comment in comments if comment not in
                pos_comments and comment not in neg_comments]


with open('NEWEST TWEETS ROHAN/compound_pos.txt', 'w', encoding="utf8") as f:
    for pos_comment in pos_comments:
        f.write('%s' % pos_comment)
with open('NEWEST TWEETS ROHAN/compound_neg.txt', 'w', encoding="utf8") as f:
    for neg_comment in neg_comments:
        f.write('%s' % neg_comment)

