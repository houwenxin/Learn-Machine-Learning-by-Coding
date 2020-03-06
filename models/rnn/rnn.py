import numpy as np
from collections import defaultdict

class RNN(object):
    def __init__(self, input_dim, output_dim, hidden_dim=32):
        '''
        Initialize a Recurrent Neural Network.
        Input: input_dim, output_dim, hidden_dim
        '''
        self.hidden_dim = hidden_dim

        # Weights: initialize with random nums
        self.W_x = np.random.randn(hidden_dim, input_dim)
        self.W_h = np.random.randn(hidden_dim, hidden_dim)
        self.W_o = np.random.randn(output_dim, hidden_dim)
        
        # Bias: initialize with 1
        self.b_h = np.zeros((hidden_dim, 1))
        self.b_o = np.zeros((output_dim, 1))

        self.grads = defaultdict(lambda: 0)

    def _tanh(self, x, derivative=False):
        '''
        Tanh function: (e^x - e^(-x)) / (e^x + e^(-x))
        derivative: return derivative
        '''
        x_safe = x + 1e-12 # To avoid small denominator
        f = (np.exp(x_safe) - np.exp(-x_safe)) / (np.exp(x_safe) + np.exp(-x_safe))
        return f if not derivative else 1 - f ** 2

    def _softmax(self, x):
        x_safe = x - np.max(x) # To prevent numerator overflow and denominator underflow.
        f = np.exp(x_safe) / (np.sum(np.exp(x_safe)))
        return f
    def _cross_entropy(self, pred, target):
        '''
        target, pred: vector of same shape (n, 1)
        '''
        return -np.mean(target * np.log(pred + 1e-12))


    def forward(self, inputs):
        '''
        inputs: a list of one-hot vector (input_dim, 1)
        '''
        h_prev = np.zeros((self.hidden_dim, 1))

        # Record hidden states & inputs for BPTT.
        self.inputs, self.hidden_states = inputs, [h_prev]

        outputs = []

        #print(self.W_x.mean())
        for t, x in enumerate(inputs):
            # forward propagation
            a = self.W_x @ x + self.W_h @ h_prev + self.b_h
            h = self._tanh(a)
            o = self.W_o @ h +self.b_o
            y = self._softmax(o)

            outputs.append(y)
            self.hidden_states.append(h)
            h_prev = h

        return outputs, self.hidden_states


    def backward(self, outputs, targets):
        loss = 0.0

        if isinstance(outputs, np.ndarray):
            #print('here')
            loss = self._cross_entropy(outputs, targets)
            d_o = outputs.copy() # Derivative of outputs.
            d_o[np.argmax(targets)] -= 1.0
            d_o_list = [1 for _ in range(len(targets) - 1)] + [d_o]
        elif isinstance(outputs, list):
            d_o_list = outputs.copy() # Derivative of outputs. shape (output_dim, 1)
            for idx, out in enumerate(outputs):
                loss += self._cross_entropy(out, targets[idx])
                d_o_list[idx][np.argmax(targets[idx])] -= 1.0
  
        # define derivatives.
        d_W_x = np.zeros_like(self.W_x)
        d_W_h = np.zeros_like(self.W_h)
        d_W_o = np.zeros_like(self.W_o)
        d_b_h = np.zeros_like(self.b_h)
        d_b_o = np.zeros_like(self.b_o)
        
        #d_h_accum = np.zeros_like()
        # Time steps excluding the initial hidden state.
        T = len(self.hidden_states) - 1

        d_h_next = np.zeros_like(self.hidden_states[0])
        for t in reversed(range(T)): 
            d_o = d_o_list[t]

            d_W_o += d_o @ self.hidden_states[t+1].T
            d_b_o += d_o

            d_h = self.W_o.T @ d_o + d_h_next
            d_a = self._tanh(self.hidden_states[t+1], derivative=True) * d_h
            d_b_h += d_a
            d_W_h += d_a @ self.hidden_states[t].T
            d_W_x += d_a @ self.inputs[t].T

            d_h_next = self.W_h.T @ d_a
        
        self.grads = {"d_W_x":d_W_x, "d_W_h":d_W_h, "d_W_o": d_W_o, "d_b_h": d_b_h, "d_b_o": d_b_o}
        
        return loss    

    def update_params(self, lr=0.1):
        for grad in self.grads.keys():
            self.grads[grad] = np.clip(self.grads[grad], -1, 1, out=self.grads[grad])
        self.W_x -= lr * self.grads['d_W_x']
        self.W_h -= lr * self.grads['d_W_h']
        self.W_o -= lr * self.grads['d_W_o']
        
        self.b_h -= lr * self.grads['d_b_h']
        self.b_o -= lr * self.grads['d_b_o']

    def zero_grad(self):
        for key in self.grads.keys():
            self.grads[key] = 0.

def generate_dataset(num_sequences=100):
    """
    From https://github.com/nicklashansen/rnn_lstm_from_scratch/blob/master/RNN_LSTM_from_scratch.ipynb

    Generates a number of sequences as our dataset.
    
    Args:
     `num_sequences`: the number of sequences to be generated.
     
    Returns a list of sequences.
    """
    samples = []
    
    for _ in range(num_sequences): 
        num_tokens = np.random.randint(0, 5)
        sample = 'a ' * num_tokens + 'b ' * num_tokens
        samples.append(sample)
    return samples



if __name__ == "__main__":
    from utils import build_vocab, text2token, token2id, one_hot_seq
    samples = generate_dataset(100)

    word2id, id2word = build_vocab(samples)
    vocab_size = len(word2id)
    inputs = []
    targets = []
    
    rnn = RNN(input_dim=vocab_size, output_dim=vocab_size)

    test_input = text2token(samples[0])[:-1]
    test_target = text2token(samples[0])[1:]
    print("Test Input:", test_input)
    print("Test Target:", test_target)
    inputs = one_hot_seq(token2id(test_input, word2id), vocab_size)
    outputs, hidden_states = rnn.forward(inputs)
    test_output = [id2word[np.argmax(out)] for out in outputs]
    print("Test Output:", test_output)

    for epoch in range(1000):
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

            rnn.update_params(lr=3e-5)
            losses.append(loss)
            #print(loss)
        print(np.array(losses).mean())
    

    print("Test Input:", test_input)
    print("Test Target:", test_target)
    inputs = one_hot_seq(token2id(test_input, word2id), vocab_size)
    outputs, hidden_states = rnn.forward(inputs)
    test_output = [id2word[np.argmax(out)] for out in outputs]
    print("Test Output:", test_output)
