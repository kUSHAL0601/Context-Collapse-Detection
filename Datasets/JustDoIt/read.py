import pandas as pd
tweets = pd.read_csv("justdoit_tweets_2018_09_07_2.csv")
text = tweets.tweet_full_text
for i in text:
	print()
	print(i)
	print()
