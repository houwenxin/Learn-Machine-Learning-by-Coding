from collections import defaultdict
import numpy as np

def tokenize(text):
    tokens = text.split(' ')
    tokens.append('EOS')
    return tokens

def build_vocab(text_list):
    vocab = {}
    for idx, text in enumerate(text_list):
        tokens = tokenize(text)
        for token in tokens:
            vocab[token] = vocab.get(token, 0) + 1
    sorted_vocab = sorted(vocab.items(), key=lambda pair:pair[1], reverse=True)
    sorted_words = [pair[0] for pair in sorted_vocab]
    sorted_words.append('UNK')

    word2id = defaultdict(lambda: len(sorted_words)-1)
    id2word = defaultdict(lambda: 'UNK')

    for index, word in enumerate(sorted_words):
        word2id[word] = index
    for index, word in enumerate(sorted_words):
        id2word[index] = word
        
    return word2id, id2word

def text2token(text):
    tokens = tokenize(text)
    return tokens
def token2id(tokens, word2id):
    ids = [word2id[token] for token in tokens]
    return ids


def one_hot_encode(idx, vocab_size):
    """
    One-hot encodes a single word given its index and the size of the vocabulary.
    
    Args:
     `idx`: the index of the given word
     `vocab_size`: the size of the vocabulary
    
    Returns a 1-D numpy array of length `vocab_size`.
    """
    # Initialize the encoded array
    one_hot = np.zeros(vocab_size)
    
    # Set the appropriate element to one
    one_hot[idx] = 1.0

    return one_hot.reshape(one_hot.shape[0], 1)

def one_hot_seq(ids, vocab_size):
    # Encode each word in the sentence
    encoding = [one_hot_encode(id, vocab_size) for id in ids]
    #encoding = encoding.reshape(*encoding.shape, 1).tolist()
    return encoding