from bs4 import BeautifulSoup
from collections import defaultdict
import io
from sentiment_common import split_text

total_words = 0
words = defaultdict(lambda: 0)

soup = BeautifulSoup(io.open("../../LargeFiles/electronics/positive.review", 'r', encoding='utf-8'), 'lxml')
for text in [review_text.text for review_text in soup.find_all('review_text')]:
    for word in split_text(text):
        total_words += 1
        words[word] += 1

soup = BeautifulSoup(io.open("../../LargeFiles/electronics/negative.review", 'r', encoding='utf-8'), 'lxml')
for text in [review_text.text for review_text in soup.find_all('review_text')]:
    for word in split_text(text):
        total_words += 1
        words[word] -= 1

print "total words ", total_words

with io.open('sentiment_word_count.txt.out','w',encoding='utf8') as f:
    for word in words:
        f.write("%s %f\n" % (word, float(words[word])/float(total_words)))
