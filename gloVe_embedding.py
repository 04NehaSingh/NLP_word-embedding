
# coding: utf-8

'''
pre_req: python 3
         anaconda 
         pandas --> 0.23
         gensim --> 3.4.0
         glove-python ---> 0.1.0
         pickle
'''

##################
# Glove embedding 
##################
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from glove import Corpus, Glove
import itertools
import gensim 
import logging
import pandas as pd
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# 1. Reading data from csv file 
df = pd.read_csv("input_file.csv")
pd.options.display.max_colwidth = 50
df.head(5)
df.isnull().sum()

clean_summaries = []
for summary in df.summary_description:
    clean_summaries.append(summary)
print("Summaries are complete.")

# 2. tokenizing and puting to list
def read_input(desc):
    for content in desc:
        yield(gensim.utils.simple_preprocess(str(content).encode("utf-8")))

documents = list(read_input(clean_summaries))
logging.info ("Done reading data file")

# documents is list of list 
print(documents[:100])

# 3. Training the model
corpus = Corpus()
corpus.fit(documents, window=10)
glove = Glove(no_components=100, learning_rate=0.05)
glove.fit(corpus.matrix, epochs=400, no_threads=4, verbose=True)
glove.save('glove_embedding_400_window_10.model')

from itertools import islice
glove.add_dictionary(corpus.dictionary)
print(list(islice(corpus.dictionary, 5)))
print("\n\n total words: "+str(len(corpus.dictionary)))

# print(type(glove))
# print(type(corpus.dictionary))
import pickle
trained_data = corpus.dictionary
# glove = Glove.load('glove_embedding.model')
# print(glove)

######################
# 4. saving the model
######################
with open('output.pickle', 'wb') as handle:
    pickle.dump(trained_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
glove.most_similar('sip', 10)

#################################################
# 5. Creating test dataset for testing the model
#################################################
from collections import Counter
import string
from nltk.corpus import stopwords

def clean_intoList(listoflist):
    flattened_list = []
    for x in listoflist:
        for y in x:
            flattened_list.append(y)
    stop_words = set(stopwords.words('english'))
    li = [w for w in flattened_list if not w in stop_words]
    return li

Counter = Counter(clean_intoList(documents))
most_occur = Counter.most_common(10)

# finding most similar words for words in list
for word in ['sip', 'sipfs', 'sipvm','wwe', 'designer', 'call', 'server', 'agent', 'error']:
    print(word)
    sim = glove.most_similar(word, number=20)
    print(sim)
    print("\n")

