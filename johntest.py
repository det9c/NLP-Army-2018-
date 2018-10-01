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


PDF_DIRECTORY="C:/Users/598936/Desktop/STUFF/DATA_SCIENCE/NLP-BOOZ/test_doc/"
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
already_tokenized=False
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

  with open(os.path.join(PDF_DIRECTORY, 'sentencestest.json'), 'w') as f:
           json.dump(sent_token_corpus, f)
  with open(os.path.join(PDF_DIRECTORY, 'wordstest.json'), 'w') as f:
           json.dump(word_token_corpus, f)
else:
    sent_token_corpus = json.load(open(os.path.join(PDF_DIRECTORY, 'sentences.json'), 'r'))
    word_token_corpus = json.load(open(os.path.join(PDF_DIRECTORY, 'words.json'), 'r'))
 #    print()
