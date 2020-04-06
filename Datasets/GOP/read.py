import pandas as pd
tweets = pd.read_csv("GOP_REL_ONLY.csv")
print(tweets.columns)
print(tweets.text)
#for i in tweets.candidate:
#	print()
#	print(i)
#	print()

