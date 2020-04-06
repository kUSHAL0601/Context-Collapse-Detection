import pandas as pd
tweets = pd.read_csv("tweets.csv")
#print(tweets.columns)
#print(tweets.head())
for i in tweets.text:
	print()
	print(i)
	print()

