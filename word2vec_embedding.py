__author__='Nehas'
# coding: utf-8

'''
pre_req: python 3
         anaconda 
         pandas --> 0.23
         gensim --> 3.4.0
         pickle
'''


############################################
# 1. creating word embedding using word2vec#
############################################
#imports
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim 
import logging
import pandas as pd

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Reading data from csv file 
df = pd.read_csv("input_file.csv")
pd.options.display.max_colwidth = 50
print(df.head(2))

# some pre-processing and merging summary - description column
df = df.drop(df.columns[df.columns.str.contains('unnamed',case = False)],axis = 1)
print(list(df.columns.values))
df['summary'] = df['summary'].str.replace(',', '').str.replace('@', '').str.lower()
df['description'] = df['description'].str.replace(',', '').str.replace('@', '').str.lower()
df['summary_description'] = df['summary'].astype(str) + df['description']
print(df['summary_description'])

#tokenizing and puting to list
desc = df['summary_description']
def read_input(desc):
    for did,content in desc.iteritems():
        if (did%10000==0):
            logging.info ("read {0} reviews".format (did))
        # do some pre-processing and return a list of words for each review text
        yield(gensim.utils.simple_preprocess(str(content).encode("utf-8")))

documents = list(read_input(desc))
logging.info ("Done reading data file")

print(documents[:100])

#training the model using word2vec
model = gensim.models.Word2Vec (documents, size=150, window=10, min_count=2, workers=10)
model.train(documents,total_examples=len(documents),epochs=15)

#######################
# 2. testing the model
#######################
w1 = "sip"
model.wv.most_similar (positive=w1)

# summarize the loaded model
print(model)

# # access vector for one word
print(model['sip'])

# save model
model.save('model.bin')

#################################
# 3. Plot Word Vectors Using PCA
#################################
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
from sklearn.decomposition import PCA
from matplotlib import*

X = model[model.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)
pyplot.scatter(result[:, 0], result[:, 1])


words = list(model.wv.vocab)
for i, word in enumerate(words):
    pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
    
#################################
# plot Word Vectors using T-Sne 
#################################
def tsne_plot(model):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []

    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)
        print(labels)
    
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=250, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    pyplot.figure(figsize=(16, 16)) 
    for i in range(len(x)):
        pyplot.scatter(x[i],y[i])
        pyplot.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    pyplot.show()


tsne_plot(model)
print(model)
print(type(model))


#####################################
# 4 saving the model in pickel format
#####################################
import pickle
trained_data = model.wv.vocab
with open('word_embedding.pickle', 'wb') as handle:
    pickle.dump(trained_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

