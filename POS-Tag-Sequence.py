'''
Predict The Most Probable Sequence Of POS Tags Of A Sentence
1. Create a clean dataset
2. Assign a POS tag to each word in each sentence (each word can have multiple POS)
3. Initialise a transition matrix with dimensions n+1 x n, where n is the number of tags
4. Initialise an emission matrix with dimensions n x v, where v is the vocabulary size
5. Assign a value to the variable epsilon
6. Populate the transition matrix by dividing the (count of each pair + epsilon) by the sum of the corresponding row
7. Populate the emission matrix by dividing the (count of word tagged with POS + epsilon) by the sum of the corresponding row
8. Initialise auxiliary matrices C and D with dimensions n x k, where k is the number of words in the given sequence
9. Populate the first column of matrix C with the product of the corresponding transition and emission probabilities
10. Populate the first column of matrix D with 0
11. Populate the remaining entries of matrix C and D column by column using the formula
12. Determine the most probable sequence starting from the back by looking at matrix C
'''

train_set = {'The cat stretched':['O','N','V'],
            'Jacob stood on his tiptoes':['N','V','O','O','N'],
            'The car turned the corner':['O','N','V','O','N'],
            'Kelly twirled in circles':['N','V','O','N'],
            'She opened the door':['O','V','O','N'],
            'Aaron made a picture':['N','V','O','N'],
            'Shelly stood by the door':['N','V','O','O','N'],
            'The car stopped':['O','N','V'],
            'He turned the picture around':['O','V','O','N','O'],
            'She pictured her in the car':['O','V','O','O','O','N'],
            'The door was open':['O','N','O','O']}

test_sequence = 'she stood by the car'

import numpy as np
from nltk.stem import PorterStemmer

porter = PorterStemmer()

EPSILON = 0.01


def cleaning(data):

    all_tokens = [porter.stem(word.lower()) for line in data.keys() for word in line.split()]

    for i in data:
        data[i] = [''] + data[i]

    all_tags = [tag for line in data.values() for tag in line]

    return all_tokens, all_tags

def init_matrices(tokens,tags):
    n = len(set(tags)) - 1
    v = len(set(tokens))

    T = np.zeros((n+1,n))
    E = np.zeros((n,v))

    return T, E

def fill_matrix_T(all_tags, T):

    tags = list(set(all_tags))

    pair_count = lambda x,y: sum([y == all_tags[i+1] for i in range(len(all_tags)-1) if x == all_tags[i]])

    for i in range(len(T)):
        for j in range(len(tags)-1):
            T[i][j] = pair_count(tags[i],tags[j+1])

    for i in range(len(T)):
        T[i] = (T[i]+EPSILON)/sum(T[i])

    return T

def fill_matrix_E(all_tokens, all_tags, E):

    tokens = sorted(list(set(all_tokens)))
    tags = list(filter(lambda tag: tag != '', all_tags))

    token_to_tag = list(zip(A,B))

    for i in range(len(E)):
        for j in range(len(tokens)):
            E[i][j] = sum([x == (tokens[j], tags[i]) for x in token_to_tag])

    for i in range(len(E)):
        E[i] = (E[i]+EPSILON)/sum(E[i])

    return E

def init_aux_matrices(tags, seq):
    n = len(set(tags)) - 1
    k = len(seq.split())

    C = np.zeros((n,k))
    D = np.zeros((n,k))

    return C,D

def fill_aux_matrices(tokens, seq, T, E, C, D):

    tokens = sorted(list(set(tokens)))

    seq = seq.split()

    for i in range(len(C)):
        C[i][0] = T[0][i] * E[i][tokens.index(seq[0])]

    for j in range(1, len(C[0])):
        for i in range(len(C)):
            candidates = [E[i][tokens.index(seq[j])] * T[k][i] * C[k-1][0] for k in range(1,len(T))]
            C[i][j] = max(candidates)
            D[i][j] = np.argmax(candidates)

    return C, D

def return_seq(C, D, tags):

    tags = list(set(tags))
    tags.remove('')

    seq_POS = [''] * len(C[0])

    seq_POS[-1] = tags[C.argmax(axis=0)[-1]]

    for i in reversed(range(len(seq_POS)-1)):
        seq_POS[i] = tags[int(D.max(axis=0)[i])]

    return seq_POS


all_tokens, all_tags = cleaning(train_set)

T, E = init_matrices(all_tokens, all_tags)
C, D = init_aux_matrices(all_tags, test_sequence)

T = fill_matrix_T(all_tags, T)
E = fill_matrix_E(all_tokens, all_tags, E)

C, D = fill_aux_matrices(all_tokens, test_sequence, T, E, C, D)

print(return_seq(C, D, all_tags))
