'''
Prepare A Dataset To Be Trained On A Continuous Bag Of Words Model
1. Upload a corpus
2. Make the dataset case insensitive with list comprehension
3. Handle punctuations (replacing interrupting marks and collapsing multi-sign marks) with regex
4. Drop special characters
5. Handle numbers (replace context-specific numbers with a special token)
6. Tokenize the string with nltk.word_tokenize
7. Extract the context and center words using a for loop and yield statement
8. Create a vocabulary from the set of center words
9. Encode each center word as a column one-hot vector
10. Encode the context words around each center word as the average of all corresponding one-hot vectors
11. Return the vectors in the form of inp -> out
'''

import re 
from nltk import word_tokenize
import numpy as np

C = 2


def cleaning(corpus):
    corpus = corpus.lower()
    corpus = re.sub(r'[!,/:;?_{}~]+', ' . ', corpus)
    corpus = re.sub(r'[#$%&\*<=>@|]+', '', corpus)

    corpus = word_tokenize(corpus)

    corpus = ['<NUMBERS>' if token.isdigit() else token for token in corpus]

    return corpus

def extract(corpus, C):
    i = C

    while i < len(corpus)-C:
        context = corpus[i-C:i] + corpus[i+1:i+C]
        center = corpus[i]
        yield context, center
        i += 1

def encode_center(word, vocab):
    vector = np.zeros((len(vocab),1))
    vector[vocab.index(word)] = 1

    return vector

def encode_context(words, vocab, dct):
    vectors = [dct[word] for word in dct]
    matrix = np.concatenate((vectors), axis=1)
    vector = matrix.sum(axis=1)/len(words)

    return vector

corpus = open('phillipine_zipcodes').read()
vocab = sorted(set(corpus[C:-C]))

dct = {}
mapping = {}

for x, y in extract(corpus, C):
    dct[y] = encode_center(y, vocab)
    mapping[y] = [x]

for y, x in mapping.items():
    print(f'{encode_context(x, vocab, dct)}: {dct[y]}')
    print('')
