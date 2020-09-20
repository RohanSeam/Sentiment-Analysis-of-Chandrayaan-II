# Rohan Seam
# 8/23/2019
# Final Project
# The purpose of this script is to create vocabulary and create Bag of Words model.

import string 
import re 
from os import listdir 
from collections import Counter 
from nltk.corpus import stopwords

# load doc into memory 
def load_doc(filename): 
    # open the file as read only 
    file = open(filename, 'r', encoding="utf8") 
    # read all text 
    text = file.read() 
    # close the file 
    file.close() 
    return text

# turn a doc into clean tokens 
def clean_doc(doc): 
    # split into tokens by white space 
    tokens = doc.split() 
    # prepare regex for char filtering 
    re_punc = re.compile('[%s]' % re.escape(string.punctuation)) 
    # remove punctuation from each word 
    tokens = [re_punc.sub('', w) for w in tokens] 
    # remove remaining tokens that are not alphabetic 
    tokens = [word for word in tokens if word.isalpha()] 
    # filter out stop words 
    stop_words = set(stopwords.words('english')) 
    tokens = [w for w in tokens if not w in stop_words] 
    # filter out short tokens 
    tokens = [word for word in tokens if len(word) > 1] 
    return tokens


# save list to file 
def save_list(lines, filename): 
    data = '\n'.join(lines) 
    file = open(filename, 'w', encoding="utf8") 
    file.write(data)
    file.close()


# load doc, clean and return line of tokens 
def doc_to_line(filename, vocab): 
    # load the doc 
    doc = load_doc(filename) 
    # clean doc 
    tokens = clean_doc(doc) 
    # filter by vocab 
    tokens = [w for w in tokens if w in vocab] 
    return ' '.join(tokens)

# load and clean a dataset
def load_clean_dataset(vocab):
    # load documents
    neg = process_docs('NEWEST TWEETS ROHAN/Neg', vocab)
    pos = process_docs('NEWEST TWEETS ROHAN/Pos', vocab)
    docs = neg + pos
    # prepare labels
    labels = [0 for _ in range(len(neg))] + [1 for _ in range(len(pos))]
    return docs, labels

'''
#NOT USED AFTER CREATING VOCAB
# load doc and add to vocab 
def add_doc_to_vocab(filename, vocab): 
    # load doc 
    doc = load_doc(filename) 
    # clean doc 
    tokens = clean_doc(doc) 
    # update counts 
    vocab.update(tokens)
'''
'''

# load all docs in a directory (THIS IS USED TO CREATE VOCAB)
def process_docs(directory, vocab): 
    # walk through all files in the folder 
    for filename in listdir(directory): 
        # skip files that do not have the right extension 
        if not filename.endswith(".txt"): 
            next 
        # create the full path of the file to open 
        path = directory + '/' + filename 
        # add doc to vocab 
        add_doc_to_vocab(path, vocab)
'''

# load all docs in a directory (THIS IS USED AFTER CREATING VOCAB)
def process_docs(directory, vocab): 
    lines = list() 
    # walk through all files in the folder 
    for filename in listdir(directory):
        # skip files that do not have the right extension 
        if not filename.startswith("cv9"): 
            next 
        # create the full path of the file to open 
        path = directory + '/' + filename 
        # load and clean the doc 
        line = doc_to_line(path, vocab) 
        # add to list 
        lines.append(line) 
    return lines

'''
#(USED TO CREATE VOCAB)
# define vocab 
vocab = Counter() 
# add all docs to vocab 
process_docs('NEWEST TWEETS ROHAN/Pos', vocab) 
process_docs('NEWEST TWEETS ROHAN/Neg', vocab) 

# print the size of the vocab 
print(len(vocab)) 

# print the top words in the vocab 
print(vocab.most_common(50)) 

# keep tokens with > 1 occurrence 
min_occurrence = 1 
tokens = [k for k,c in vocab.items() if c >= min_occurrence] 
print(len(tokens)) 

# save tokens to a vocabulary file 
save_list(tokens, 'NEWEST TWEETS ROHAN/vocab.txt')


'''
# load vocabulary (USED AFTER VOCAB CREATED)
vocab_filename = 'NEWEST TWEETS ROHAN/vocab.txt' 
vocab = load_doc(vocab_filename) 
vocab = vocab.split() 
vocab = set(vocab) 

# load all training reviews
docs, labels = load_clean_dataset(vocab)
# summarize what we have
print(len(docs), len(labels))


#(USED TO CREATE NEGATIVE.txt and POSITIVE.txt)
# prepare negative reviews 
negative_lines = process_docs('NEWEST TWEETS ROHAN/Neg', vocab) 
save_list(negative_lines, 'NEWEST TWEETS ROHAN/negative.txt') 

# prepare positive reviews 
positive_lines = process_docs('NEWEST TWEETS ROHAN/Pos', vocab) 
save_list(positive_lines, 'NEWEST TWEETS ROHAN/positive.txt')
