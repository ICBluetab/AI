from bs4 import BeautifulSoup
from collections import defaultdict
import io
import pandas as pd
import numpy as np

words = defaultdict(lambda: 0)

with io.open("sentiment_word_count.txt.out", 'r', encoding='utf-8') as f:
    for line in f:
        split = line.split(' ')
        words[split[0]] = float(split[1])



words_len = len(words)
keys = words.keys()
df = pd.DataFrame(columns = keys + ['T'])

soup = BeautifulSoup(io.open("../../LargeFiles/electronics/negative.review", 'r', encoding='utf-8'), 'lxml')
for text in [review_text.text for review_text in soup.find_all('review_text')]:
    df.loc[df.shape[0]] = [0] * words_len + [0]
    for word in text.split():
        if word in keys:
            df.loc[df.shape[0] - 1][word] = words[word]

soup = BeautifulSoup(io.open("../../LargeFiles/electronics/positive.review", 'r', encoding='utf-8'), 'lxml')
for text in [review_text.text for review_text in soup.find_all('review_text')]:
    df.loc[df.shape[0]] = [0] * words_len + [1]
    for word in text.split():
        if word in keys:
            df.loc[df.shape[0] - 1][word] = words[word]


df.to_csv("sentiment_words.csv.out", encoding='utf-8')
