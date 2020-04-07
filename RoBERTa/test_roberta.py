import pandas as pd

def get_coachella():
	with open("../Datasets/Coachella/context.txt") as f:
		context=f.readlines()[0].strip().replace('(',' ').replace(')',' ')
	f.close()
	LABELS = {'negative': 0, 'cant tell': 1, 'positive': 2,'neutral': 1}
	tweets = pd.read_csv("../Datasets/Coachella/Coachella-2015-2-DFE.csv")

	Y=tweets.coachella_sentiment.map(LABELS)
	txt=tweets.text

	Y=Y.values
	right = txt.values
	X = []
	for i in right:
		X.append([context,i])
	return X, Y


def get_gop():
	with open("../Datasets/GOP/context.txt") as f:
		context=f.readlines()[0].strip().replace('(',' ').replace(')',' ')
	f.close()
	LABELS = {'Negative': 0, 'Positive': 2,'Neutral': 1}
	tweets = pd.read_csv("../Datasets/GOP/GOP_REL_ONLY.csv")

	Y=tweets.sentiment.map(LABELS)

	txt=tweets.text
	Y=Y.values

	right = txt.values
	X = []
	for i in right:
		X.append([context,i])
	return X, Y

import torch
from fairseq.data.data_utils import collate_tokens

roberta = torch.hub.load('pytorch/fairseq', 'roberta.large.mnli')
roberta.eval()

print('Loaded Roberta')

X,Y=get_coachella()
print('Loaded Coachella')

X1,Y1=get_gop()
print('Loaded GOP')

X=collate_tokens(
    [roberta.encode(pair[0], pair[1]) for pair in X], pad_idx=1
)
print('Tokenized Coachella')

X1=collate_tokens(
    [roberta.encode(pair[0], pair[1]) for pair in X1], pad_idx=1
)
print('Tokenized GOP')

import numpy as np

Y=np.array(Y[:100])
Yp=np.array(roberta.predict('mnli', X[:100,:]).argmax(dim=1))
print('Predicted Coachella')

Y1=np.array(Y1[:100])
Yp1=np.array(roberta.predict('mnli', X1[:100,:]).argmax(dim=1))
print('Predicted GOP')

print(Yp)
print()
print(Yp1)

print((Y==Yp).sum()/len(Y))
print((Y1==Yp1).sum()/len(Y1))