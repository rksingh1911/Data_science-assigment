# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 18:56:09 2021

@author: Hi
"""

#============ TOKENIZATION USING SPACY  =======================================

import spacy
#nlp = spacy.load('en_core_web_sm')
nlp = spacy.load("en_core_web_sm")

doc2 = nlp("Tesla isn't   looking into startups anymore.")

for token in doc2:
    print(token.text)
    
doc2
doc2[0]

type(doc2)

# Part-of-Speech Tagging (POS)
doc2[0].pos_

#Dependencies
doc2[0].dep_
spacy.explain('PROPN')

spacy.explain('nsubj')

#Spans
doc3 = nlp(u'Although commmonly attributed to John Lennon from his song "Beautiful Boy", \
the phrase "Life is what happens to us while we are making other plans" was written by \
cartoonist Allen Saunders and published in Reader\'s Digest in 1957, when Lennon was 17.')

for token in doc3:
    print(token.text)


life_quote = doc3[16:30]
print(life_quote)

type(life_quote)

#Sentences
doc4 = nlp('This is the first sentence. This is another sentence. This is the last sentence.')

for sent in doc4.sents:
    print(sent)
    
doc4[6].is_sent_start



# Create a string that includes opening and closing quotation marks

# Create a Doc object and explore tokens
doc = nlp(u"We are moving to L.A.!")

for token in doc:
    print(token.text)

for token in doc:
    print(token.text, end=' | ')
    
    
doc5 = nlp(u"We're here to help! Send snail-mail, email support@oursite.com or visit us at http://www.oursite.com!")

for t in doc5:
    print(t)
# comment: spacy has its internal rules what can be tokens assigned at where
#    it is realizing that websites and email ID's and hence it not broken in to 
#    tokenss
    
doc6 = nlp(u'A 5km NYC cab ride costs $10.30')

for t in doc6:
    print(t)
    
# comment: $ is separated and the values 10.30 is not separated
    
#Exceptions
doc7 = nlp(u"Let's visit St. Louis in the U.S. next year.")

for t in doc7:
    print(t)

# comment: here the country name is tokenized with fullstops.

#Counting Tokens
len(doc7)

#Counting Vocab Entries
# len(doc.vocab)
# comment: it will based on the language library what you have used at start


#Tokens can be retrieved by index position and slice

doc8 = nlp(u'It is better to give than to receive.')

# Retrieve the third token:
doc8[2]


# Retrieve three tokens from the middle:
doc8[2:5]

# Retrieve the last four tokens:
doc8[-4:]

# comment --> Looks like a list but tokens can never re-assignment the values/strings
doc8[2] = 'word' # error
     

#==============================================

doc9 = nlp(u'Apple to build a Hong Kong factory for $6 million')

  
for token in doc9:
    print(token.text, end='|')

for token in doc9:
    print(token.text, end=' | ')

for entity in doc9.ents:
    print(entity)
    
# comment: the most popular words and it relavent it will trace out from the string
# named entity recogination
for entity in doc9.ents:
    print(entity)
    print(entity.label_)
    print("\n")

#ORG--> orgnaizaiton
    # 6 million --> money
    
for entity in doc9.ents:
    print(entity)
    print(entity.label_)
    print(str(spacy.explain(entity.label_)))
    print("\n")


#==============================================================================
    
doc10 = nlp(u"Autonomous cars shift insurance liability toward manufacturers.")

for chunk in doc10.noun_chunks:
    print(chunk)
    
#==============================================================================


from spacy import displacy

doc = nlp("This is a sentence.")
displacy.serve(doc, style="dep")
# 127.0.0.1:5000


doc13 = nlp(u'Apple is going to build a U.K. factory for $6 million.')
displacy.serve(doc13, style="dep")

    
# 127.0.0.1:5000
    


#============ TOKENIZATION USING NLTK  =======================================

#import nltk
#nltk.download()

from nltk.tokenize import word_tokenize
text = "Hello, How are you? Are you available to come to interview?"
print(word_tokenize(text))

from nltk.tokenize import sent_tokenize
text = "God is Great! I won a lottery."
print(sent_tokenize(text))




