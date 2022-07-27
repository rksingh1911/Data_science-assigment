# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 19:01:21 2021

@author: Hi
"""
#==============================================================================    
#Stemming
# Import the toolkit and the full Porter Stemmer library
# import nltk
# nltk.download()

from nltk.stem.porter import PorterStemmer
p_stemmer = PorterStemmer()
words = ['run','runner','running','ran','runs','easily','fairly']

for word in words:
    print(word+' --> '+p_stemmer.stem(word))


from nltk.stem import PorterStemmer
p_stemmer = PorterStemmer()
e_words= ["wait", "waiting", "waited", "waits"]

for word in e_words:
    print(word+' --> '+p_stemmer.stem(word))

#============================================================================== 
#Snowball Stemmer
from nltk.stem.snowball import SnowballStemmer

# The Snowball Stemmer requires that you pass a language parameter
s_stemmer = SnowballStemmer(language='english')

words = ['run','runner','running','ran','runs','easily','fairly']

for word in words:
    print(word+' --> '+s_stemmer.stem(word))
    

words2 = ['generous','generation','generously','generate']


#============================================================================== 
for word in words2:
    print("p_stemmer - ",word+' --> '+s_stemmer.stem(word))

for word in words2:
    print("s_stemmer - ",word+' --> '+p_stemmer.stem(word))


#============================================================================== 
phrase = 'I will meets him tomorrow at the meeting'
phrase.split()

for word in phrase.split():
    print(word+' --> '+p_stemmer.stem(word))









words = ['consolingly']

print('Porter Stemmer:')
for word in words:
    print(word+' --> '+p_stemmer.stem(word))
    
print('Porter2 Stemmer:')
for word in words:
    print(word+' --> '+s_stemmer.stem(word))
    
phrase = 'I am meeting him tomorrow at the meeting'
for word in phrase.split():
    print(word+' --> '+p_stemmer.stem(word))
    

