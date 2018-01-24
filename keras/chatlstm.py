'''
Created on Jan 2, 2017
@author: camilo

Notes:

	- without weighting, the network learns a majority-class classifier
	- with weighthing, the model tends to be very precise (~ 1) for one class
	  and has very high recall (~ 1) for the other, with F1-score close to 0.5
	- performance improves slightly when using a LSTM layer (0.55 F1-score)
	- precision, recall and F1 show great variability across runs
	- numbers are more reliable when the validation set is balanced
	- the training set is very small (141 texts), which explains the not so good results

'''


from __future__ import print_function


import os
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding, LSTM
from keras.models import Model, Sequential
import json
from pandas import DataFrame
from sklearn.metrics import classification_report
from keras import backend as K
import gzip


# mute tensor flow library warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


class BuildCorpus(object):
    '''
    class to build training
    set for classification, extract
    BOWs features and train an SVM classifier
    '''
    
    raw_data    = None # JSON data    
    full_data   = None # full frame with TFIDF BOWs          
    

    def __init__(self,jsonpath):
        '''
        sample data
        '''
        json_data=open(jsonpath).read().decode('utf-8').encode('utf-8', 'ignore')
        self.raw_data = json.loads(json_data)
        self.build()   
        
        
    def build(self):
        '''
        build training/testing/cross-validation set
        '''
        rows    = []
        index   = []
        '''
        loop over JSON objects
        '''
        cnt = 0
        for item in self.raw_data:
            '''
            populate data frame
            '''            
            if item.get('level'):
                text    = item['description']
                tag     = item['level']
                ind     = item['title']
                cnt = cnt + 1
                rows.append({'description': text + ind, 'level': tag, 'title': ind})
                index.append(cnt)
                
        '''
        create and save data frame
        '''
        data_frame = DataFrame(rows, index)
        self.full_data  = data_frame


BASE_DIR = '/home/camilo'
W2V_DIR  = os.path.join(BASE_DIR + '/nlp-semantics/Word2Vec')


MAX_SEQUENCE_LENGTH = 10000
MAX_NUM_WORDS       = 4200
EMBEDDING_DIM       = 50
VALIDATION_SPLIT    = 0.2


# first, build index mapping words in the embeddings set
# to their embedding vector

print('--------------------------------------------------------------')
print('indexing word vectors')


embeddings_index = {}
f = open(W2V_DIR + '/glove.6B.50d.txt', 'r')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='string')
    embeddings_index[word] = coefs
f.close()

print('found %s word vectors' % len(embeddings_index))


# second, prepare text samples and their labels
print('--------------------------------------------------------------')
print('processing text dataset')


dataset         = BuildCorpus('/home/camilo/workspace-FIN/keraschat/data/data.json')
df              = dataset.full_data
# list of text samples
texts           = [s.encode('utf-8') for s in df.description.tolist()]
# dictionary mapping label name to numeric id
labels_index    = {"Senior Level":0, "Entry Level":1, "Mid Level":1, "Internship":1}
labels_simp     = {"Senior Level":0, "Other":1}
# list of label ids 
labels          = [labels_index[s] for s in df.level.tolist()]


# finally, vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)


word_index = tokenizer.word_index
print('found %s unique tokens.' % len(word_index))


data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)


labels = to_categorical(np.asarray(labels))
print('shape of data tensor:', data.shape)
print('shape of label tensor:', labels.shape)


# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])


x_train = data[:-num_validation_samples]
y_train = labels[:-num_validation_samples]
x_val = data[-num_validation_samples:]
y_val = labels[-num_validation_samples:]


def myunique(array):
    '''
    compute distribution of labels
    encoded as one-hot vectors
    '''
    res = {}
    for a in array:
        if str(a) not in res.keys():
            res[str(a)]=1
        else:
            res[str(a)]=res[str(a)]+1
    return res


print('--------------------------------------------------------------')
print('training set class distribution:')     
print('--------------------------------------------------------------')
sdict1 = myunique(y_train)
print(sdict1)       
print('training set size', len(y_train)) 
print('--------------------------------------------------------------')
print('test set class ditribution:')
print('--------------------------------------------------------------')
sdict2 = myunique(y_val)
print(sdict2)    
print('test set size', len(y_val))

print('--------------------------------------------------------------')
print('preparing embedding matrix')


# prepare embedding matrix
num_words = min(MAX_NUM_WORDS, len(word_index)+1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= MAX_NUM_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector


# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)


print('training model')


# train a 1D convnet with global maxpooling
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='float32') 	# input layer
embedded_sequences = embedding_layer(sequence_input)			# embedding layer
x = Conv1D(50, 5, activation='relu')(embedded_sequences)		# convolution layer
x = MaxPooling1D(5)(x)							# pooling layer
x = Conv1D(50, 5, activation='relu')(x)					# convolution layer
x = MaxPooling1D(5)(x)							# pooling layer
# (un)comment to switch betweeen LSTM and pure CN
x = LSTM(128)(x)								# lstm layer
# (un)comment to switch betweeen LSTM and pure CNN
#x = Conv1D(128, 5, activation='relu')(x)				# convolution layer
#x = GlobalMaxPooling1D()(x)						# pooling layer
x = Dense(128, activation='relu')(x)					# dense deep layer
preds = Dense(len(labels_simp), activation='softmax')(x)		# output layer
model = Model(sequence_input, preds)


# train
model.compile(loss='binary_crossentropy',
              	optimizer='sgd',
		metrics=['accuracy']
              )


# training data is unbalanced (50 vs 75)
sdict3  = myunique(y_train)
tot 	= len(y_train)
weights = {0:1.0, 1:1.0}
for k in sdict3.keys():
	if k == '[ 0.  1.]':
		weights[0] = float(sdict3[k])/float(tot)
	else:
		weights[1] = float(sdict3[k])/float(tot)


print('class weighthing:', weights)
print('--------------------------------------------------------------')


# cross-validation
model.fit(x_train, y_train,
          batch_size=128,
          epochs=5,
          class_weight = weights,
          validation_data=(x_val, y_val)
          )


# predictions and probabilities
probs    = model.predict(x_val)
labs     = probs.argmax(axis=-1)
cmap	 = {0:'Senior Level', 1:'Other'}
preds    = ['(' + cmap[lab] + ', ' + `max(p)` + ')' for lab,p in zip(labs.tolist(),probs)]


# classification report (evaluation)
print('--------------------------------------------------------------')
print('model performance (F1, precision, recall):')
labs = to_categorical(labs.astype('float32'), 2) # coerce to categorical
print('--------------------------------------------------------------\n')
print(classification_report(y_val, labs))
print('--------------------------------------------------------------')


# sample predictions
print('top K predictions:')
print('--------------------------------------------------------------')
print('\n'.join(preds[0:10]))
print('--------------------------------------------------------------')
print('label mappings:')
print('--------------------------------------------------------------')
print(labels_simp)
print('--------------------------------------------------------------')


# save model to file
print("saving model to disk!")
try:
	json_string = model.to_json()
	open('/home/camilo/workspace-FIN/keraschat/data/chat-bin-model.json', 'w').write(json_string) # save model
	model.save_weights('/home/camilo/workspace-FIN/keraschat/data/chat-bin-model.h5') # save weights			
except: 
	Exception
print('--------------------------------------------------------------')


# model stats
model.summary() 
