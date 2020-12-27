'''
Autocomplete A Sentence Given The Start Of The Sentence
1. Create a corpus and list of prompts
2. Clean the corpus
3. Calculate the probabilities of each bigram
4. Use the probability matrix to complete the sentence
'''

corpus = open('short_stories').read()[:1000]

prompts = ['maybe it was',
          'this was',
          'she']

from string import punctuation
import numpy as np
import re

def clean(corpus):
    corpus = re.findall(r"(\w+'?\w+|\.)", corpus)

    new_corpus = []

    for word in corpus:
        if word == '.':
            new_corpus += ['</s>','<s>']
        else:
            new_corpus.append(word)

    return new_corpus

def P_matrix(vocab, corpus):
    C = np.zeros((len(vocab),len(vocab)))

    bigram_count = lambda x,y: sum(y == corpus[i+1] for i in range(len(corpus)-1) if x == corpus[i])

    for i in range(len(C)):
        for j in range(len(C)):
            C[i][j] = bigram_count(vocab[i],vocab[j])

    P = C+1
    P = P/P.sum(axis=1)

    return P

def complete(sentence, vocab, P):
    sentence = sentence.split()

    word = sentence[-1]

    while word != '</s>' and len(sentence) < 10:
        i = vocab.index(word)
        word = vocab[P[i].argmax()]
        sentence.append(word)

    return ' '.join(sentence)

new_corpus = clean(corpus)

vocab = sorted({word: new_corpus.count(word) for word in new_corpus})

P = P_matrix(vocab, new_corpus)

for sentence in prompts:
    print(sentence)
    print(complete(sentence, vocab, P))
    print('')
