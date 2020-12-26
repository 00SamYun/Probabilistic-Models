'''
Use Autocorrect To Edit A Paragraph - Scaled Down
1. Set up a simple dictionary of words and corpus
2. Compare each word in the input with the dictionary and flag misspelled words
3. Set a maximum n edit distance
4. Find all words in the dictionary that are <= n edit distance away from the misspelled word
5. Calculate the word probabilities of each word candidate using the corpus
6. Return the candidate with the highest probability
'''

import numpy as np
from english_words import english_words_lower_alpha_set

DICTIONARY = english_words_lower_alpha_set
INSERT, DELETE, REPLACE = 1,1,2
CORPUS = open('simple_text_corpus','r').read()

def flag(text):
    flagged = []
    for word in text.split():
        if word not in DICTIONARY:
            flagged.append(word)
    return flagged

def min_edit_d(source,target):
    rows = len(source)+1
    cols = len(target)+1
    table = np.zeros((rows,cols))
    for x in range(1,rows):
        table[x][0] = table[x-1][0]+DELETE
    for x in range(1,cols):
        table[0][x] = table[0][x-1]+INSERT
    for i in range(1,rows):
        for j in range(1,cols):
            table[i][j] = min([table[i-1][j] + DELETE,
                              table[i][j-1] + INSERT,
                              table[i-1][j-1] + [REPLACE if source[i-1] != target[j-1] else 0][0]])
    return table[-1][-1]

def find_candidates(word):
    candidates = []
    for i in DICTIONARY:
        if min_edit_d(word,i) <= n:
            candidates.append(i)
    return candidates

def get_prob(candidate):
    return CORPUS.split().count(candidate)/len(CORPUS.split())

def autocorrect(text):
    n = 3
    inps = flag(text)
    for inp in inps:
        candidates = find_candidates(inp)
        max_prob = 0
        out = ''
        for candidate in candidates:
            if get_prob(candidate) > max_prob:
                out = candidate
        text = text.replace(inp,out)

    return text

# example:
# autocorrect('what is your ides of guilty plesure')
# this will output: 'what is your idea of guilty pleasure'
