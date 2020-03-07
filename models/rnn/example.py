from rnn import RNN
import os
import pandas as pd
import numpy as np
from utils import build_vocab, text2token, token2id, one_hot_seq

def prepare_data():
    os.system('wget https://www.kaggle.com/c/17777/download-all')
    os.system('unzip nlp-getting-started.zip')
    os.system('rm nlp-getting-started.zip')

def real_test():
    if not os.path.exists('train.csv'):
        prepare_data()

    data = pd.read_csv('train.csv')
    texts = data['text'].tolist()
    labels = data['target'].tolist()

    word2id, id2word = build_vocab(texts)
    vocab_size = len(word2id)

    ids_list = []
    for text in texts:
        ids_list.append(token2id(text2token(text), word2id))
    
    num_class = len(set(labels))
    rnn = RNN(input_dim=vocab_size, output_dim=num_class, hidden_dim=256)

    accs = []
    ids_list = ids_list[:100]
    labels = labels[:100]
    for epoch in range(10):
        losses = []
        for inputs, label in zip(ids_list, labels):
            inputs = one_hot_seq(inputs, vocab_size)
            label = one_hot_seq([label], num_class)[0]
            outputs, hidden_states = rnn.forward(inputs)
            #print(rnn.grads['d_W_x'])
            pred = outputs[-1]
            rnn.zero_grad()
            loss = rnn.backward(pred, label)
            accs.append(np.argmax(label) == np.argmax(pred))
            rnn.update_params(lr=2e-4)
            losses.append(loss)
            #print(loss)
        print("Epoch", epoch, "Loss:", np.array(losses).mean(), "Acc:", np.array(accs).mean())


def generate_dataset(num_sequences=100):
    """
    From https://github.com/nicklashansen/rnn_lstm_from_scratch/blob/master/RNN_LSTM_from_scratch.ipynb

    Generates a number of sequences as our dataset.
    
    Args:
     `num_sequences`: the number of sequences to be generated.
     
    Returns a list of sequences.
    """
    import random
    samples = []
    
    for _ in range(num_sequences): 
        num_tokens = np.random.randint(3, 10)
        sample = 'a ' * num_tokens + 'b ' * num_tokens
        samples.append(sample[:-1])
    return samples

def sim_test():

    samples = generate_dataset(100)

    word2id, id2word = build_vocab(samples)
    vocab_size = len(word2id)
    inputs = []
    targets = []
    
    rnn = RNN(input_dim=vocab_size, output_dim=vocab_size, hidden_dim=256)

    test_input = text2token(samples[0])[:-1]
    test_target = text2token(samples[0])[1:]
    print("Test Input:", test_input)
    print("Test Target:", test_target)
    inputs = one_hot_seq(token2id(test_input, word2id), vocab_size)
    outputs, hidden_states = rnn.forward(inputs)
    test_output = [id2word[np.argmax(out)] for out in outputs]
    print("Test Output:", test_output)

    for epoch in range(150):
        losses = []
        for sample in samples:
            ids = token2id(text2token(sample), word2id)
            inputs = one_hot_seq(ids[:-1], vocab_size)
            targets = one_hot_seq(ids[1:], vocab_size)
            #print(inputs[0].shape)
            
            outputs, hidden_states = rnn.forward(inputs)
            #print(rnn.grads['d_W_x'])
            rnn.zero_grad()
            loss = rnn.backward(outputs, targets)

            rnn.update_params(lr=2e-4)
            losses.append(loss)
            #print(loss)
        print(np.array(losses).mean())
    
    print("Test Input:", test_input)
    print("Test Target:", test_target)
    inputs = one_hot_seq(token2id(test_input, word2id), vocab_size)
    outputs, hidden_states = rnn.forward(inputs)
    test_output = [id2word[np.argmax(out)] for out in outputs]
    print("Test Output:", test_output)

if __name__ == "__main__":
    sim_test()
    

