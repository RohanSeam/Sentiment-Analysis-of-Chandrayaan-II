# Rohan Seam
# 8/23/2019
# Final Project
# The purpose of this script is to simulate a multichannel model.

'''
import string 
import re 
from os import listdir 
from nltk.corpus import stopwords 
from pickle import dump

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
    tokens = ' '.join(tokens) 
    return tokens

# load all docs in a directory
def process_docs(directory, is_train):
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
        tokens = clean_doc(doc)
        # add to list
        documents.append(tokens)
    return documents

# load and clean a dataset
def load_clean_dataset(is_train):
    # load documents
    neg = process_docs('NEWEST TWEETS ROHAN/Neg', is_train)
    pos = process_docs('NEWEST TWEETS ROHAN/Pos', is_train)
    docs = neg + pos
    # prepare labels
    labels = [0 for _ in range(len(neg))] + [1 for _ in range(len(pos))]
    return docs, labels

# save a dataset to file 
def save_dataset(dataset, filename): 
    dump(dataset, open(filename, 'wb')) 
    print('Saved: %s' % filename)
    
# load and clean all reviews 
train_docs, ytrain = load_clean_dataset(True) 
test_docs, ytest = load_clean_dataset(False) 

# save training datasets 
save_dataset([train_docs, ytrain], 'NEWEST TWEETS ROHAN/train.pkl') 
save_dataset([test_docs, ytest], 'NEWEST TWEETS ROHAN/test.pkl')
'''
'''

from pickle import load 
from numpy import array 
from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences 
from keras.utils.vis_utils import plot_model 
from keras.models import Model 
from keras.layers import Input 
from keras.layers import Dense 
from keras.layers import Flatten 
from keras.layers import Dropout 
from keras.layers import Embedding 
from keras.layers.convolutional import Conv1D 
from keras.layers.convolutional import MaxPooling1D 
from keras.layers.merge import concatenate

# load a clean dataset 
def load_dataset(filename): 
    return load(open(filename, 'rb'))

# fit a tokenizer 
def create_tokenizer(lines): 
    tokenizer = Tokenizer() 
    tokenizer.fit_on_texts(lines) 
    return tokenizer

# calculate the maximum document length 
def max_length(lines): 
    return max([len(s.split()) for s in lines])

# encode a list of lines 
def encode_text(tokenizer, lines, length): 
    # integer encode 
    encoded = tokenizer.texts_to_sequences(lines) 
    # pad encoded sequences 
    padded = pad_sequences(encoded, maxlen=length, padding='post') 
    return padded

# define the model 
def define_model(length, vocab_size): 
    # channel 1 
    inputs1 = Input(shape=(length,)) 
    embedding1 = Embedding(vocab_size, 100)(inputs1) 
    conv1 = Conv1D(filters=32, kernel_size=4, activation='relu')(embedding1) 
    drop1 = Dropout(0.5)(conv1) 
    pool1 = MaxPooling1D(pool_size=2)(drop1) 
    flat1 = Flatten()(pool1) 
    
    # channel 2 
    inputs2 = Input(shape=(length,)) 
    embedding2 = Embedding(vocab_size, 100)(inputs2) 
    conv2 = Conv1D(filters=32, kernel_size=6, activation='relu')(embedding2) 
    drop2 = Dropout(0.5)(conv2) 
    pool2 = MaxPooling1D(pool_size=2)(drop2) 
    flat2 = Flatten()(pool2) 
    
    # channel 3 
    inputs3 = Input(shape=(length,)) 
    embedding3 = Embedding(vocab_size, 100)(inputs3) 
    conv3 = Conv1D(filters=32, kernel_size=8, activation='relu')(embedding3) 
    drop3 = Dropout(0.5)(conv3) 
    pool3 = MaxPooling1D(pool_size=2)(drop3) 
    flat3 = Flatten()(pool3) 
    
    # merge 
    merged = concatenate([flat1, flat2, flat3]) 
    # interpretation 
    dense1 = Dense(10, activation='relu')(merged) 
    outputs = Dense(1, activation='sigmoid')(dense1) 
    model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs) 
    
    # compile 
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 
    
    # summarize 
    model.summary() 
    plot_model(model, show_shapes=True, to_file='model4.png') 
    return model

# load training dataset 
trainLines, trainLabels = load_dataset('NEWEST TWEETS ROHAN/train.pkl') 

# create tokenizer 
tokenizer = create_tokenizer(trainLines) 

# calculate max document 
length = max_length(trainLines) 
print('Max document length: %d' % length) 

# calculate vocabulary size 
vocab_size = len(tokenizer.word_index) + 1 
print('Vocabulary size: %d' % vocab_size) 

# encode data 
trainX = encode_text(tokenizer, trainLines, length) 

# define model 
model = define_model(length, vocab_size) 

# fit model 
model.fit([trainX,trainX,trainX], array(trainLabels), epochs=7, batch_size=16) 

# save the model 
model.save('NEWEST TWEETS ROHAN/model2.h5')
'''


import re
import string
from pickle import load
from numpy import array
from nltk.corpus import stopwords 
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

# load a clean dataset
def load_dataset(filename):
    return load(open(filename, 'rb'))

# fit a tokenizer
def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

# Clean document of stopwords and punctuation
def clean_doc(doc):
    tokens = doc.split()
    # regex prep for character filtering
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    tokens = [re_punc.sub('', w) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    #filter out short tokens
    tokens = [word for word in tokens if len(word) > 1]
    tokens = ' '.join(tokens)
    return tokens

# calculate the maximum document length
def max_length(lines):
    return max([len(s.split()) for s in lines])

# encode a list of lines
def encode_text(tokenizer, lines, length):
    # integer encode
    encoded = tokenizer.texts_to_sequences(lines)
    # pad encoded sequences
    padded = pad_sequences(encoded, maxlen=length, padding='post')
    return padded

def predict_sentiment(review, tokenizer, max_length, model):
    line = clean_doc(review)
    padded = encode_text(tokenizer, [line], max_length)
    # Since three channel, must input padded vectors as list of three
    yhat = model.predict([padded, padded, padded], verbose=0)
    percent_pos = yhat[0,0]
    if round(percent_pos) == 0:
        return (1-percent_pos), 'NEGATIVE'
    return percent_pos, 'POSITIVE'

# load datasets
trainLines, trainLabels = load_dataset('NEWEST TWEETS ROHAN/train.pkl')
testLines, testLabels = load_dataset('NEWEST TWEETS ROHAN/test.pkl')

# create tokenizer
tokenizer = create_tokenizer(trainLines)

# calculate max document length
length = max_length(trainLines)
print('Max document length: %d' % length)

# calculate vocabulary size
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary size: %d' % vocab_size)

# encode data
trainX = encode_text(tokenizer, trainLines, length)
testX = encode_text(tokenizer, testLines, length)

# load the model
model = load_model('NEWEST TWEETS ROHAN/model2.h5')

# evaluate model on training dataset
_, acc = model.evaluate([trainX,trainX,trainX], array(trainLabels), verbose=0)
print('Train Accuracy: %.2f' % (acc*100))

# evaluate model on test dataset dataset
_, acc = model.evaluate([testX,testX,testX], array(testLabels), verbose=0)
print('Test Accuracy: %.2f' % (acc*100))

file = open('NEWEST TWEETS ROHAN/test.txt', 'r')
output = open('NEWEST TWEETS ROHAN/multichannelNetworkSentiment.txt', 'w')
line = file.readline()
while line:
    percent, sentiment = predict_sentiment(line, tokenizer, length, model)
    output.write('Review: [%s]\nSentiment: %s (%.3f%%)\n' % (line, sentiment, percent*100))
    line = file.readline()
file.close()
output.close()
