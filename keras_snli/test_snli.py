from __future__ import print_function
from functools import reduce
import json
import os
import re
import tarfile
import tempfile

import numpy as np
np.random.seed(1234)  # for reproducibility

'''
300D Model - Train / Test (epochs)
=-=-=
Batch size = 512
Fixed GloVe
- 300D SumRNN + Translate + 3 MLP (1.2 million parameters) - 0.8315 / 0.8235 / 0.8249 (22 epochs)
- 300D GRU + Translate + 3 MLP (1.7 million parameters) - 0.8431 / 0.8303 / 0.8233 (17 epochs)
- 300D LSTM + Translate + 3 MLP (1.9 million parameters) - 0.8551 / 0.8286 / 0.8229 (23 epochs)

Following Liu et al. 2016, I don't update the GloVe embeddings during training.
Unlike Liu et al. 2016, I don't initialize out of vocabulary embeddings randomly and instead leave them zeroed.

The jokingly named SumRNN (summation of word embeddings) is 10-11x faster than the GRU or LSTM.

Original numbers for sum / LSTM from Bowman et al. '15 and Bowman et al. '16
=-=-=
100D Sum + GloVe - 0.793 / 0.753
100D LSTM + GloVe - 0.848 / 0.776
300D LSTM + GloVe - 0.839 / 0.806
'''

import keras
import keras.backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import merge, recurrent, Dense, Input, Dropout, TimeDistributed
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import Bidirectional
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.regularizers import l2
from keras.utils import np_utils
import pandas as pd

def get_coachella():
	# with open("../Datasets/Coachella/dataset.txt") as f:
	# 	lines=f.readlines()
	# f.close()
	with open("../Datasets/Coachella/context.txt") as f:
		context=f.readlines()[0].strip().replace('(',' ').replace(')',' ')
	f.close()
	# # print(context)
	# right=[]
	# for i in lines:
	# 	if i and i!='\n':
	# 		right.append(i.strip())
	# left=context*len(right)
	# with open("../Datasets/Coachella/labels.txt") as f:
	# 	labels=f.readlines()
	# f.close()
	# # print(labels[:10])
	LABELS = {'negative': 0, 'cant tell': 3, 'positive': 2,'neutral': 1}
	# Y=[]
	# for i in labels:
	# 	if i and i!='\n':
	# 		Y.append(LABELS[i.strip()])
	# print(len(left),len(right),len(Y))
	# Y = np_utils.to_categorical(np.array(Y), 3)
	tweets = pd.read_csv("../Datasets/Coachella/Coachella-2015-2-DFE.csv")
	# print(tweets.columns)
	Y=tweets.coachella_sentiment.map(LABELS)
	txt=tweets.text
	Y=Y.values
	Y = np_utils.to_categorical(np.array(Y), 4)
	right = txt.values
	left = []
	for i in range(3846):
		left.append(context)
	# print(context)
	return left, right, Y


def get_gop():
	# with open("../Datasets/Coachella/dataset.txt") as f:
	# 	lines=f.readlines()
	# f.close()
	with open("../Datasets/GOP/context.txt") as f:
		context=f.readlines()[0].strip().replace('(',' ').replace(')',' ')
	f.close()
	# # print(context)
	# right=[]
	# for i in lines:
	# 	if i and i!='\n':
	# 		right.append(i.strip())
	# left=context*len(right)
	# with open("../Datasets/Coachella/labels.txt") as f:
	# 	labels=f.readlines()
	# f.close()
	# # print(labels[:10])
	LABELS = {'Negative': 0, 'Positive': 2,'Neutral': 1}
	# Y=[]
	# for i in labels:
	# 	if i and i!='\n':
	# 		Y.append(LABELS[i.strip()])
	# print(len(left),len(right),len(Y))
	# Y = np_utils.to_categorical(np.array(Y), 3)
	tweets = pd.read_csv("../Datasets/GOP/GOP_REL_ONLY.csv")
	# print(tweets.columns)
	Y=tweets.sentiment.map(LABELS)
	txt=tweets.text
	Y=Y.values
	Y = np_utils.to_categorical(np.array(Y), 3)
	right = txt.values
	left = []
	for i in range(len(right)):
		left.append(context)
	# print(context)
	return left, right, Y


from keras.models import load_model

MAX_LEN = 230

tokenizer = Tokenizer(lower=False, filters='')

# Lowest index from the tokenizer is 1 - we need to include 0 in our vocab count
VOCAB = len(tokenizer.word_counts) + 1


to_seq = lambda X: pad_sequences(tokenizer.texts_to_sequences(X), maxlen=MAX_LEN)
prepare_data = lambda data: [to_seq(data[0]), to_seq(data[1]), data[2]]


data = get_gop()
tokenizer.fit_on_texts(data[0] + data[1])
data = prepare_data(data)


l=len(data[0])

idx=[i for i in range(l)]

from random import shuffle

shuffle(idx)


train_idx=idx[:int(l*0.7)]
val_idx=idx[int(l*0.7):int(l*0.7)+int(l*0.2)]
test_idx=idx[int(l*0.7)+int(l*0.2):]

train = [[],[],[]]
val = [[],[],[]]
test = [[],[],[]]

for i in train_idx:
	train[0].append(np.array(data[0][i]))
	train[1].append(np.array(data[1][i]))
	train[2].append(np.array(data[2][i]))

for i in val_idx:
	val[0].append(np.array(data[0][i]))
	val[1].append(np.array(data[1][i]))
	val[2].append(np.array(data[2][i]))

for i in test_idx:
	test[0].append(np.array(data[0][i]))
	test[1].append(np.array(data[1][i]))
	test[2].append(np.array(data[2][i]))


# model = load_model('model.h5')
# loss, acc = model.evaluate([test[0], test[1]], test[2], batch_size=BATCH_SIZE)
# print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))




VOCAB = len(tokenizer.word_counts) + 1
# LABELS = {'contradiction': 0, 'neutral': 1, 'entailment': 2}
LABELS = {'negative': 0, 'positive': 2,'neutral': 1}

# RNN = recurrent.LSTM
RNN = lambda *args, **kwargs: Bidirectional(recurrent.LSTM(*args, **kwargs))
# RNN = recurrent.GRU
#RNN = lambda *args, **kwargs: Bidirectional(recurrent.GRU(*args, **kwargs))
# Summation of word embeddings
# RNN = None
LAYERS = 3
USE_GLOVE = False
TRAIN_EMBED = True
EMBED_HIDDEN_SIZE = 300
SENT_HIDDEN_SIZE = 300
BATCH_SIZE = 64
PATIENCE = 4 # 8
MAX_EPOCHS = 10
MAX_LEN = 230
DP = 0.2
L2 = 4e-6
ACTIVATION = 'relu'
OPTIMIZER = 'rmsprop'
print('RNN / Embed / Sent = {}, {}, {}'.format(RNN, EMBED_HIDDEN_SIZE, SENT_HIDDEN_SIZE))
print('GloVe / Trainable Word Embeddings = {}, {}'.format(USE_GLOVE, TRAIN_EMBED))

print('Build model...')
print('Vocab size =', VOCAB)

GLOVE_STORE = 'precomputed_glove.weights'
if USE_GLOVE:
  if not os.path.exists(GLOVE_STORE + '.npy'):
    print('Computing GloVe')
  
    embeddings_index = {}
    f = open('glove.840B.300d.txt')
    for line in f:
      values = line.split(' ')
      word = values[0]
      coefs = np.asarray(values[1:], dtype='float32')
      embeddings_index[word] = coefs
    f.close()
    
    # prepare embedding matrix
    embedding_matrix = np.zeros((VOCAB, EMBED_HIDDEN_SIZE))
    for word, i in tokenizer.word_index.items():
      embedding_vector = embeddings_index.get(word)
      if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
      else:
        print('Missing from GloVe: {}'.format(word))
  
    np.save(GLOVE_STORE, embedding_matrix)

  print('Loading GloVe')
  embedding_matrix = np.load(GLOVE_STORE + '.npy')

  print('Total number of null word embeddings:')
  print(np.sum(np.sum(embedding_matrix, axis=1) == 0))

  embed = Embedding(VOCAB, EMBED_HIDDEN_SIZE, weights=[embedding_matrix], input_length=MAX_LEN, trainable=TRAIN_EMBED)
else:
  embed = Embedding(VOCAB, EMBED_HIDDEN_SIZE, input_length=MAX_LEN)

rnn_kwargs = dict(output_dim=SENT_HIDDEN_SIZE, dropout_W=DP, dropout_U=DP)
SumEmbeddings = keras.layers.core.Lambda(lambda x: K.sum(x, axis=1), output_shape=(SENT_HIDDEN_SIZE, ))

translate = TimeDistributed(Dense(SENT_HIDDEN_SIZE, activation=ACTIVATION))

premise = Input(shape=(MAX_LEN,), dtype='int32')
hypothesis = Input(shape=(MAX_LEN,), dtype='int32')

prem = embed(premise)
hypo = embed(hypothesis)

prem = translate(prem)
hypo = translate(hypo)

if RNN and LAYERS > 1:
  for l in range(LAYERS - 1):
    rnn = RNN(return_sequences=True, **rnn_kwargs)
    prem = BatchNormalization()(rnn(prem))
    hypo = BatchNormalization()(rnn(hypo))
rnn = SumEmbeddings if not RNN else RNN(return_sequences=False, **rnn_kwargs)
prem = rnn(prem)
hypo = rnn(hypo)
prem = BatchNormalization()(prem)
hypo = BatchNormalization()(hypo)

# joint = merge([prem, hypo], mode='concat')
joint = keras.layers.merge.concatenate([prem, hypo])
joint = Dropout(DP)(joint)
for i in range(4):
  joint = Dense(2 * SENT_HIDDEN_SIZE, activation=ACTIVATION, W_regularizer=l2(L2) if L2 else None)(joint)
  joint = Dropout(DP)(joint)
  joint = BatchNormalization()(joint)

pred = Dense(len(LABELS), activation='softmax')(joint)

model = Model(input=[premise, hypothesis], output=pred)
model.compile(optimizer=OPTIMIZER, loss='categorical_crossentropy', metrics=['accuracy'])
																																																																																					
model.summary()


print('Training')																																																																																																																																																																																																																														
_, tmpfn = tempfile.mkstemp()
# Save the best model during validation and bail out of training early if we're not improving
callbacks = [EarlyStopping(patience=PATIENCE), ModelCheckpoint(tmpfn, save_best_only=True, save_weights_only=True)]
model.fit([train[0], train[1]], np.array(train[2]), batch_size=BATCH_SIZE, nb_epoch=MAX_EPOCHS, validation_data=([val[0], val[1]], np.array(val[2])), callbacks=callbacks)

model.save("gop_model_bilstm_230.h5")

# Restore the best found model during validation
model.load_weights(tmpfn)

loss, acc = model.evaluate([test[0], test[1]], np.array(test[2]), batch_size=BATCH_SIZE)
print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))
