# Libraries 
import spacy
from scipy import spatial


# Load English Library
nlp = spacy.load('en_core_web_md')
print(nlp.pipe_names)


# Size of Vocabulary
print(len(nlp.vocab))
print(len(nlp.vocab.vectors))


# Dimensions in Our Vectors
print(nlp(u'lion').vector.shape)


# Checking Some Similarities
tokens = nlp(u'cat lion pet')

for t1 in tokens:
    for t2 in tokens:
        print(t1.text, t2.text, t1.similarity(t2))


# Just Checking Some Tokens
tokens = nlp(u'dog cat alex kyniv')
for t in tokens:
    print(t.text,t.has_vector,t.vector_norm,t.is_oov)


# Creating Vectors of wolf, wild, pet
wolf = nlp(u'wolf').vector
wild = nlp(u'wild').vector
pet = nlp(u'pet').vector


# Performing Pperation on The Vectors
new_vector = wolf - wild + pet
print(new_vector)


# Creating a cosine similarity function
cosine_similarity = lambda vec1,vec2 : 1-spatial.distance.cosine(vec1,vec2)


# Checking The Similarity of Our New Vector With Every Word in Our New Vocab
similarities = []

for word in nlp.vocab:
    if word.has_vector and word.is_alpha and word.is_lower:
        similarities.append((cosine_similarity(new_vector,word.vector),word.text))


# Printing The Top 20 Similar Words
for similarity, word in sorted(similarities, reverse= True) [:20]:
    print(word)