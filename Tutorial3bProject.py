# Rohan Seam
# 8/23/2019
# Final Project
# The purpose of this script is to simulate a CNN model. 

'''
import string 
import re 
from os import listdir 
from numpy import array 
from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences 
from keras.utils.vis_utils import plot_model 
from keras.models import Sequential 
from keras.layers import Dense 
from keras.layers import Flatten 
from keras.layers import Embedding 
from keras.layers.convolutional import Conv1D 
from keras.layers.convolutional import MaxPooling1D

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
def clean_doc(doc, vocab): 
    # split into tokens by white space 
    tokens = doc.split() 
    # prepare regex for char filtering 
    re_punc = re.compile('[%s]' % re.escape(string.punctuation)) 
    # remove punctuation from each word 
    tokens = [re_punc.sub('', w) for w in tokens] 
    # filter out tokens not in vocab 
    tokens = [w for w in tokens if w in vocab] 
    tokens = ' '.join(tokens) 
    return tokens

# load all docs in a directory 
def process_docs(directory, vocab, is_train): 
    documents = list() 
    # walk through all files in the folder 
    for filename in listdir(directory): 
        # skip any reviews in the test set 
        if is_train and filename.startswith('cv9'): 
            continue 
        if not is_train and not filename.startswith('cv9'): 
            continue 
        # create the full path of the file to open 
        path = directory + '/' + filename 
        # load the doc 
        doc = load_doc(path)
        # clean doc 
        tokens = clean_doc(doc, vocab) 
        # add to list 
        documents.append(tokens) 
    return documents

# load and clean a dataset 
def load_clean_dataset(vocab, is_train): 
    # load documents 
    neg = process_docs('NEWEST TWEETS ROHAN/Neg', vocab, is_train) 
    pos = process_docs('NEWEST TWEETS ROHAN/Pos', vocab, is_train) 
    docs = neg + pos 
    # prepare labels 
    labels = array([0 for _ in range(len(neg))] + [1 for _ in range(len(pos))]) 
    return docs, labels

# fit a tokenizer 
def create_tokenizer(lines): 
    tokenizer = Tokenizer() 
    tokenizer.fit_on_texts(lines) 
    return tokenizer

# integer encode and pad documents 
def encode_docs(tokenizer, max_length, docs): 
    # integer encode 
    encoded = tokenizer.texts_to_sequences(docs) 
    # pad sequences 
    padded = pad_sequences(encoded, maxlen=max_length, padding='post') 
    return padded

# define the model 
def define_model(vocab_size, max_length): 
    model = Sequential() 
    model.add(Embedding(vocab_size, 100, input_length=max_length)) 
    model.add(Conv1D(filters=32, kernel_size=8, activation='relu')) 
    model.add(MaxPooling1D(pool_size=2)) 
    model.add(Flatten()) 
    model.add(Dense(10, activation='relu')) 
    model.add(Dense(1, activation='sigmoid')) 
    # compile network 
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 
    # summarize defined model 
    model.summary() 
    plot_model(model, to_file='model3.png', show_shapes=True) 
    return model




'''
'''
# load the document 
filename = 'pos/compound_pos.txt' 
text = load_doc(filename) 
tokens = clean_doc(text) 
print(tokens) 
'''
'''

# load the vocabulary 
vocab_filename = 'NEWEST TWEETS ROHAN/vocab.txt' 
vocab = load_doc(vocab_filename) 
vocab = set(vocab.split()) 

# load training data 
train_docs, ytrain = load_clean_dataset(vocab, True) 

# create the tokenizer 
tokenizer = create_tokenizer(train_docs) 

# define vocabulary size 
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary size: %d' % vocab_size) 

# calculate the maximum sequence length 
max_length = max([len(s.split()) for s in train_docs])
print('Maximum length: %d' % max_length) 

# encode data 
Xtrain = encode_docs(tokenizer, max_length, train_docs) 
# define model 
model = define_model(vocab_size, max_length) 
# fit network 
model.fit(Xtrain, ytrain, epochs=10, verbose=2) 

# save the model 
model.save('NEWEST TWEETS ROHAN/model.h5')
'''


import string
import re
from os import listdir
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

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
def clean_doc(doc, vocab):
    # split into tokens by white space
    tokens = doc.split()
    # prepare regex for char filtering
    re_punc = re.compile( '[%s]' % re.escape(string.punctuation))
    # remove punctuation from each word
    tokens = [re_punc.sub( '', w) for w in tokens]
    # filter out tokens not in vocab
    tokens = [w for w in tokens if w in vocab]
    tokens = ' '.join(tokens)
    return tokens


# load all docs in a directory
def process_docs(directory, vocab, is_train):
    documents = list()
    # walk through all files in the folder
    for filename in listdir(directory):
        # skip any reviews in the test set
        if is_train and filename.startswith('cv9'):
            continue
        if not is_train and not filename.startswith('cv9'):
            continue
        # create the full path of the file to open
        path = directory + '/' + filename
        # load the doc
        doc = load_doc(path)
        # clean doc
        tokens = clean_doc(doc, vocab)
        # add to list
        documents.append(tokens)
    return documents

# load and clean a dataset
def load_clean_dataset(vocab, is_train):
    # load documents
    neg = process_docs('NEWEST TWEETS ROHAN/Neg', vocab, is_train)
    pos = process_docs('NEWEST TWEETS ROHAN/Pos', vocab, is_train)
    docs = neg + pos
    # prepare labels
    labels = array([0 for _ in range(len(neg))] + [1 for _ in range(len(pos))])
    return docs, labels


# fit a tokenizer
def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

# integer encode and pad documents
def encode_docs(tokenizer, max_length, docs):
    # integer encode
    encoded = tokenizer.texts_to_sequences(docs)
    # pad sequences
    padded = pad_sequences(encoded, maxlen=max_length, padding='post')
    return padded

# classify a review as negative or positive
def predict_sentiment(review, vocab, tokenizer, max_length, model):
    # clean review
    line = clean_doc(review, vocab)
    # encode and pad review
    padded = encode_docs(tokenizer, max_length, [line])
    # predict sentiment
    yhat = model.predict(padded, verbose=0)
    # retrieve predicted percentage and label
    percent_pos = yhat[0,0]
    if round(percent_pos) == 0:
        return (1-percent_pos), 'NEGATIVE'
    return percent_pos, 'POSITIVE'

# load the vocabulary
vocab_filename = 'NEWEST TWEETS ROHAN/vocab.txt'
vocab = load_doc(vocab_filename)
vocab = set(vocab.split())

# load all reviews
train_docs, ytrain = load_clean_dataset(vocab, True)
test_docs, ytest = load_clean_dataset(vocab, False)

# create the tokenizer
tokenizer = create_tokenizer(train_docs)

# define vocabulary size
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary size: %d' % vocab_size)
# calculate the maximum sequence length
max_length = max([len(s.split()) for s in train_docs])
print('Maximum length: %d' % max_length)

# encode data
Xtrain = encode_docs(tokenizer, max_length, train_docs)
Xtest = encode_docs(tokenizer, max_length, test_docs)

# load the model
model = load_model('NEWEST TWEETS ROHAN/model.h5')

# evaluate model on training dataset
_, acc = model.evaluate(Xtrain, ytrain, verbose=0)
print('Train Accuracy: %.2f' % (acc*100))

# evaluate model on test dataset
_, acc = model.evaluate(Xtest, ytest, verbose=0)
print('Test Accuracy: %.2f' % (acc*100))

cleanTweets  = open("NEWEST TWEETS ROHAN/test.txt" , "r")
with open('NEWEST TWEETS ROHAN/Tweet_Sentiments_CNN.txt', 'w') as readmefile:

	for line in cleanTweets:
		percent, sentiment = predict_sentiment(line, vocab, tokenizer,max_length, model)
		readmefile.write('%sSentiment: %s (%.3f%%)' % (line, sentiment, percent*100) + '\n')

cleanTweets.close()