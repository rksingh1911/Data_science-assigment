"""
Created on Fri Mar 26 13:23:50 2021
spaCy Basics
spaCy (https://spacy.io/) is an open-source Python library that parses and "understands" large volumes of text. 
Separate models are available that cater to specific languages (English, French, German, etc.).

In this section we'll install and setup spaCy to work with Python, and then 
introduce some concepts related to Natural Language Processing.
"""

# Run in anaconda prompt, before run make sure you need to exit from user
conda install -c conda-forge spacy
# After running about command it will take minimum 10 minutes of time

# to cross check it is downloaded or not
# python -m spacy download en
python -m spacy download en_core_web_sm

# Import spaCy and load the language library
import spacy
nlp = spacy.load("en_core_web_sm")
# en --> core english langugae, web_sm --> it is small version 


#=================================================================
# Create a Doc object
#  u is a unicode string is for indicaton text message
doc = nlp(u'Tesla is looking at buying U.S. startup for $6 million')

# using the language library we are going to parse this entire stream in to  separate components for us
# and its going to parse into tokens essentially.
# Print each token separately
for token in doc:
    print(token.text)

# Spacy will also realize that U.S is country name, $ is us dollar, 6 is another value

for token in doc:
    print(token.text, token.pos_)
# token.pos_ --> parts of speech 

'''
for token in doc:
    print(token.text, token.pos_, token.dep_)
#token.dep_  --> Syntactic dependency
'''

nlp.pipeline
# This steps will shows you the series of operations taggin, parsing, describing

nlp.pipe_names # displays the names what are possible

#=================================================================
doc2 = nlp(u"Tesla isn't looking into startups anymore.")

for token in doc2:
    print(token.text)


for token in doc2:
    print(token.text, token.pos_)

len(doc2)
doc2[0]
doc2[1]
doc2[1]

doc2[0].pos_

print(doc2[0].pos_)
print(doc2[0].text)
print(doc2[0].lemma_)
print(doc2[0].tag_)
print(doc2[0].shape_)
print(doc2[0].is_alpha)
print(doc2[0].is_stop)

#=================================================================
doc3 = nlp(u'Although commmonly attributed to John Lennon from his song "Beautiful Boy", \
the phrase "Life is what happens to us while we are making other plans" was written by \
cartoonist Allen Saunders and published in Reader\'s Digest in 1957, when Lennon was 17.')

# i like to span some words instead of all

life_quote = doc3[16:30]
life_quote
type(life_quote)
type(doc3)



#=================================================================
# sentence tokenization
doc4 = nlp(u'This is the first sentence. This is another sentence. This is the last sentence.')
doc4

doc4.sents

for sent in doc4.sents:
    print(sent)
    
doc4[1]
doc4[6]

# it is checking that sentence is 
doc4[6].is_sent_start 

# no response means it is not starting
doc4[7].is_sent_start 


# Note: spacy is such powerfull library, when we just pass a raw string 
# it is capable of understanding parts of speech, named entity recognation
# sentence starts and ends and many more
