'''
Calculate The Probability Of A Sentence Based On A Given Corpus
1. Create a corpus and testing dataset
2. Process the corpus
3. Create a vocabulary
4. Create a count matrix
5. Transform the count matrix into a probability matrix
6. Apply laplacian smoothing
7. Process the testing dataset
8. Use the probability matrix and the formula to calculate the probability of the sentence
'''
corpus = ['I love to wait',
         'Joe waited for the train',
         'The train was late',
         'Mary and Samantha took the bus',
         'I looked for Mary and Samantha at the bus station',
         'Mary and Samantha arrived at the bus station early but waited until noon for the bus',
         'The cat stretched',
         'Jacob stood on his tiptoes',
         'The car turned the corner',
         'Kelly danced in circles',
         'She opened the door',
         'Aaron made a picture',
         'I am sorry',
         'I danced at the store',
         'Sarah and her mother drove to the store',
         'Jenny and I opened all the gifts',
         'The cat and dog ate',
         'My mother and I went to a movie',
         'Mrs. Juarez and Mr. Smith are dancing gracefully',
         'Samantha, Elizabeth, and Joan are on the committee',
         'The ham, green beans, mashed potatoes, and corn are gluten-free',
         'The paper and pencil sat idle on the desk',
         'Misha walked and danced around',
         'My mother hemmed and hawed over where to go for dinner',
         'He was eating and talking',
         'I rinsed and dried the dishes',
         'Joe stood up and spoke to the crowd']

test_set = ['Joe went to the cat and dog store',
            'Sarah and Jessie are eating dinner',
            'The frog danced and waited in the store',
            'The dinner smells delicious',
            'There is a cat in the car with mother',
            'I am out of paper for the printer',
            'The music is too loud for my ears']

from nltk import PorterStemmer
import numpy as np

porter = PorterStemmer()


def process(corpus):
    MIN_FREQ = 2

    corpus = ['<s> ' + line + ' </s>' for line in corpus]
    corpus = [porter.stem(word.lower()) for line in a for word in line.split()]

    vocab = {word: corpus.count(word) for word in corpus}
    vocab = dict(filter(lambda item: item[1] >= MIN_FREQ, vocab.items()))

    corpus = list(map(lambda i:'UNK' if corpus[i] not in vocab else corpus[i], range(len(corpus))))

    vocab['UNK'] = corpus.count('UNK')

    return corpus, vocab

def C_matrix(vocab, corpus):
    words = sorted(vocab)
    C = np.zeros((len(vocab),len(vocab)))

    bigram_count = lambda x,y: sum(y == corpus[i+1] for i in range(len(corpus)-1) if x == corpus[i])

    for i in range(len(C)):
        for j in range(len(C)):
            C[i][j] = bigram_count(words[i],words[j])

    return C

def P_matrix(C):
    P = C+1

    P = P/P.sum(axis=1)

    return P

def process_test(sentence):
    sentence = list(map(lambda word: porter.stem(word.lower()), sentence.split()))

    sentence = list(map(lambda word: 'UNK' if word not in vocab else word, sentence))

    sentence = ['<s>'] + sentence + ['</s>']

    return sentence

def get_prob(sentence, vocab, P):
    probability = 1

    for i,word in enumerate(sentence[:-1]):
        probability *= P[sorted(vocab).index(word)][sorted(vocab).index(sentence[i+1])]

    return probability

corpus, vocab = process(corpus)

C = C_matrix(vocab, corpus)
P = P_matrix(C)

for test in test_set:
    prob = get_prob(process_test(test), vocab, P)
    print(test, ': {}'.format(prob))
