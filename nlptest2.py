from __future__ import division, print_function

import PyPDF2
import sys
import os
import numpy as np
import nltk
import re
import pprint
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.corpus import stopwords
import string
import json
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer 
from nltk.stem import PorterStemmer
from sklearn.cross_validation import train_test_split


PDF_DIRECTORY="C:/Users/598936/Desktop/STUFF/DATA_SCIENCE/NLP-BOOZ/Archive/"
REGEX               = re.compile('[^a-zA-Z]+', re.UNICODE)
STOP_WORDS = stopwords.words('english')
PUNCTUATIONS        = [str(p) for p in string.punctuation] #unicode is str in Python v3
MIN_SENTENCE_LENGTH = 5
MIN_WORD_LENGTH = 2

def load_pdf(path): #Function aken from Robert Milletich demo.py code
    """Loads pdf pages into one string

    Parameters
    ----------
    path : str
        Path to pdf file

    Returns
    -------
    text : str
        Text in pdf concatenated into one string
    """
    # Open pdf object
    pdf    = open(path, 'rb')
    reader = PyPDF2.PdfFileReader(pdf)

    # Combine all text
    text, count = "", 0
    while count < reader.numPages:
        tmp    = reader.getPage(count)
        count += 1
        text  += tmp.extractText()

    return text



file_names = [f for f in os.listdir(PDF_DIRECTORY) if f.endswith('.pdf')]
num_documents=len(file_names)
word_token_corpus=[]
sent_token_corpus=[]
icount=0
already_tokenized=True
if(not already_tokenized):
  for this_file in file_names:
      icount+=1
      print("Processing file "+str(icount)+" of "+str(num_documents))
      this_file=load_pdf(PDF_DIRECTORY+this_file)
      scratch_list=word_tokenize(this_file)
      good_words = [REGEX.sub("", word.lower()) for word in scratch_list if not word in STOP_WORDS and not word in PUNCTUATIONS and len(word)>MIN_WORD_LENGTH]
      scratch_list = []
      for k in range(0,len(good_words)):
         if(good_words[k] != ''):
             scratch_list.append(good_words[k])
      word_token_corpus.append(scratch_list)
      scratch_list=nltk.sent_tokenize(this_file)
      scratch_list2=[]
      for sentence in scratch_list:
         scratch_vec=sentence.split()
         good_words = [REGEX.sub("", word.lower()) for word in scratch_vec if not word in STOP_WORDS and not word in PUNCTUATIONS and len(word)>MIN_WORD_LENGTH]
         tmp=[]
         for k in range(0,len(good_words)):
#             if(good_words[k] != '' and len(good_words[k])>0):
             if(len(good_words[k])>0):
                  tmp.append(good_words[k])
         if(len(tmp)>0):         
            scratch_list2.append(tmp)
            
      sent_token_corpus.append(scratch_list2)    #WILL HAVE TO FILTER BY SENTENCE LENGTH LATER TO MATCH RM CODE

  with open(os.path.join(PDF_DIRECTORY, 'sentences.json'), 'w') as f:
           json.dump(sent_token_corpus, f)
  with open(os.path.join(PDF_DIRECTORY, 'words.json'), 'w') as f:
           json.dump(word_token_corpus, f)
else:
    sent_token_corpus = json.load(open(os.path.join(PDF_DIRECTORY, 'sentences.json'), 'r'))
    word_token_corpus = json.load(open(os.path.join(PDF_DIRECTORY, 'words.json'), 'r'))
 #    print()
    
documents=[]
ps=PorterStemmer()
'''for idoc in range(0,num_documents):
    paper=""
    print(idoc)
    for this_word in word_token_corpus[idoc]:
        paper=paper+" "+ps.stem(this_word)
    documents.append(paper)    
'''
#with open(os.path.join(PDF_DIRECTORY, 'documentxxx.json'), 'w') as f:
#    json.dump(documents, f)
documents = json.load(open(os.path.join(PDF_DIRECTORY, 'documentxxx.json'), 'r'))    

np.random.seed(10)
ind=np.linspace(0,len(documents)-1,len(documents))
train, test,i_train,i_test =train_test_split(documents,ind,test_size=.05)



#flat_list = [item for sublist in word_token_corpus for item in sublist]
#vectorizer_words=TfidfVectorizer()
#vectorizer_words=CountVectorizer()
#X=vectorizer_words.fit_transform(documents)
#vectorizer_words.get_feature_names()
#bag_word=X.toarray()   
#km = KMeans(n_clusters=5, init='k-means++', max_iter=100, n_init=1,)
#km.fit(X)
n_samples = 20000
n_features = 1500
n_components = 10
n_top_words = 12



vectorizer_words=CountVectorizer(max_features=n_features,stop_words='english',ngram_range=(1,3))
tf=vectorizer_words.fit_transform(train)
lda = LatentDirichletAllocation(n_components=n_components, max_iter=5,learning_method='online',learning_offset=50.,random_state=10)
lda.fit(tf)
tf_feature_names = vectorizer_words.get_feature_names()

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()
print("LDA WITH WORD COUNT")
print_top_words(lda, tf_feature_names, n_top_words)




tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,max_features=n_features,stop_words='english',ngram_range=(1,3))
tfidf = tfidf_vectorizer.fit_transform(train)
nmf = NMF(n_components=n_components,alpha=.1, l1_ratio=.5,random_state=10).fit(tfidf)
tfidf_feature_names = tfidf_vectorizer.get_feature_names()
print("nNMF WITH TFIDF")
print_top_words(nmf, tfidf_feature_names, n_top_words)



# x1=X[:,0].A convert sparse to dense using .A
#lda = LatentDirichletAllocation(n_components=10, max_iter=5)
#lda.fit(X)


n_components=20
print("PCA WITH WORD COUNT")
lsi_model = TruncatedSVD(n_components=n_components,random_state=10)
lsi_Z = lsi_model.fit_transform(tf)
print_top_words(lsi_model, tf_feature_names, n_top_words)
p1=lsi_model.singular_values_


print("PCA WITH TFIDF")
lsi_model = TruncatedSVD(n_components=n_components,random_state=10)
lsi_Z = lsi_model.fit_transform(tfidf)
print_top_words(lsi_model, tfidf_feature_names, n_top_words)
p2=lsi_model.singular_values_


tfidf2 = tfidf_vectorizer.transform(test)
lsi_Z = lsi_model.transform(tfidf2)
k=len(test)
num_categories=2
for i in range (0,k):
    print("Document "+str(i)+" "+str(file_names[int(i_test[i])]))
    projections=np.abs(lsi_Z[i])
    projections=projections/np.sqrt(np.sum(projections*projections))
    indices=np.argsort(-projections)
    for j in range(0,num_categories):
        print("Topic "+str(indices[j]))




PDF_DIRECTORY="C:/Users/598936/Desktop/STUFF/DATA_SCIENCE/NLP-BOOZ/test_doc/"
word_token_corpus = json.load(open(os.path.join(PDF_DIRECTORY, 'wordstest.json'), 'r'))
 #    print()

documents=[]
ps=PorterStemmer()
for idoc in range(0,1):
    paper=""
    print(idoc)
    for this_word in word_token_corpus[idoc]:
        paper=paper+" "+ps.stem(this_word)
    documents.append(paper)


tfidf2 = tfidf_vectorizer.transform(documents)
lsi_Z = lsi_model.transform(tfidf2)
k=1
num_categories=4
for i in range (0,k):
    print("Document "+str(i))
    projections=np.abs(lsi_Z[i])
    projections=projections/np.sqrt(np.sum(projections*projections))
    indices=np.argsort(-projections)
    for j in range(0,num_categories):
        print("Topic "+str(indices[j]))


    







        '''
a=np.linspace(0,n_components-1,n_components)
plt.subplot(1,2,1)
plt.plot(a,p1,label='WORD_COUNT')
plt.legend(loc='upper right')
plt.subplot(1,2,2)
plt.plot(a,p2,label='TF-IDF')
plt.legend(loc='upper right')
plt.show()
'''



