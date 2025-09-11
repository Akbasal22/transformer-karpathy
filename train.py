import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import random


# open input.txt
f = open('input.txt', 'r', encoding='utf-8')
text = f.read()
chars = sorted(list(set(text)))


#global variables
block_size = 8
batch_size = 4
vocab_size = len(chars)
n_embed = 32
head_size = 16
dropout = 0.2
num_heads = 8
epochs = 5000
eval_interval = 500
n_layers = 8
learning_rate = 3e-4



##I will improve this with regex helped byte pair encoding,
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
def encode(text):
    return [stoi[char] for char in text]
def decode(tokens):
    text = "".join(itos[token] for token in tokens)
    return text


#split data
data = np.array(encode(text))
train_set = data[:9*len(data)//10]  
test_set = data[9*len(data)//10:]  


def get_batch(data):
    ix = [random.randint(0, len(data) - block_size) for _ in range(batch_size)]
    x = [data[i:i+block_size] for i in ix]
    y = [data[i+1:i+1+block_size] for i in ix]
    return np.array(x), np.array(y)


#single head of attention
class Head(tf.keras.layers.Layer):
    def __init__(self, head_size, dropout=0.0):
        super().__init__()
        self.key = tf.keras.layers.Dense(units=head_size,use_bias=False)
        self.query = tf.keras.layers.Dense(units=head_size,use_bias=False)
        self.value = tf.keras.layers.Dense(units=head_size,use_bias=False)
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x):
        T = tf.shape(x)[1] #WHAT IS STATIC VS DYNAMIC SHAPE
        #create k and q matrices
        k = self.key(x)
        q = self.query(x)
        #calculate q@k.T
        wei = tf.matmul(q, k, transpose_b=True) * (head_size ** -0.5)
        #mask upper triangle indices for softmax
        mask = tf.linalg.band_part(tf.ones((T, T)), -1, 0)
        wei = tf.where(mask == 0, tf.fill(tf.shape(wei), -float('inf')), wei)
        #softmax
        wei = tf.nn.softmax(wei, axis=-1)
        wei = self.dropout(wei)
        #calculate wei@v
        v = self.value(x)
        out = tf.matmul(wei, v)
        return out
    
        #???? nn.dropout



#multiple heads of attention
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads= [Head(head_size, dropout) for _ in range(num_heads)]
        self.proj = tf.keras.layers.Dense(n_embed)
        self.dropout = tf.keras.layers.Dropout(dropout)
    def call(self,x):
        out = tf.concat([h(x) for h in self.heads], axis=-1)
        out = self.proj(self.dropout(out))
        return out


#feed forward (multilayer perceptron)
class MLP(tf.keras.layers.Layer):
    def __init__(self, n_embd):
        super().__init__()
        self.net = tf.keras.Sequential([
            tf.keras.layers.Dense(units=4*n_embd),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(units=n_embd),
            tf.keras.layers.Dropout(dropout)
        ])

    def call(self, x):
        return self.net(x)

#single block consisting of (multiple attention + MLP)
class Block(tf.keras.layers.Layer):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa =MultiHeadAttention(n_head, head_size)
        self.mlp = MLP(n_embd)

        #layernorm
        self.ln1 = tf.keras.layers.LayerNormalization()
        self.ln2 = tf.keras.layers.LayerNormalization()

    def call(self, x):
        x = x + self.sa(self.ln1(x))
        x += x + self.mlp(self.ln2(x))
        return x
    

class Transformer(tf.keras.Model):
    def __init__(self):
        super().__init__()
        #I should initalize these two in tensorflow layer type
        self.token_embedding_table =tf.keras.layers.Embedding(input_dim = vocab_size, output_dim = n_embed)
        self.position_embedding_table = tf.keras.layers.Embedding(input_dim = block_size, output_dim = n_embed)


        self.blocks = [Block(n_embed, num_heads) for _ in range(n_layers)]
        self.ln_f = tf.keras.layers.LayerNormalization()
        
        #what is this self.lm_head?
        self.lm_head = tf.keras.layers.Dense(units=vocab_size)

        #how to initialize weights?

    #what is this idx parameter in forward method
    def call(self, idx, targets = None):
        T = tf.shape(idx)[1]
        tok_emb = self.token_embedding_table(idx)
        positions = tf.range(start=0, limit=T, dtype=tf.int32) #???? explain
        pos_emb = self.position_embedding_table(positions)
        pos_emb = tf.expand_dims(pos_emb, axis=0)
        x = tok_emb + pos_emb


        for block in self.blocks:
            x = block(x)
        # why not directly x = self.blocks(x)


        x = self.ln_f(x)
        logits = self.lm_head(x)
        

        #what does this do?
        # if not targets:
        #     loss = None
        # else :
        #     flat_logits = tf.reshape(logits, (-1, self.vocab_size))
        #     flat_targets = tf.reshape(targets, (-1,))
        #     scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
        #     per_example_loss = scce(flat_targets, flat_logits)  # (B*T,)
        #     loss = tf.reduce_mean(per_example_loss)

        return logits


    #copy pasted from gpt have no idea what it does
    # optional generation method (greedy sampling with optional top_k)
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        idx: tensor (B, T) initial context
        returns: tensor (B, T + max_new_tokens) with generated tokens appended
        """
        for _ in range(max_new_tokens):
            # crop context to block_size
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self.call(idx_cond, training=False)  # (B, T', vocab)
            logits = logits[:, -1, :]  # (B, vocab) take last time step
            logits = logits / (tf.cast(temperature, tf.float32) + 1e-9)

            if top_k is not None:
                top_k = int(top_k)
                values, _ = tf.math.top_k(logits, k=top_k)
                min_values = tf.reduce_min(values, axis=-1, keepdims=True)
                logits = tf.where(logits < min_values, -1e9, logits)
 
            next_token = tf.random.categorical(logits, num_samples=1)  # (B, 1)
            idx = tf.concat([idx, tf.cast(next_token, idx.dtype)], axis=1)
        return idx
    

#is this necessary? cant I just directly use numpy arrays for datasets???
def make_dataset(data, block_size, batch_size, shuffle=True):
    # Create overlapping sequences of length block_size+1
    ds = tf.data.Dataset.from_tensor_slices(data)
    ds = ds.window(block_size+1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(block_size+1))

    # Split into inputs ([:-1]) and targets ([1:])
    ds = ds.map(lambda window: (window[:-1], window[1:]))

    if shuffle:
        ds = ds.shuffle(10000)

    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = make_dataset(train_set, block_size, batch_size, shuffle=True)
val_ds   = make_dataset(test_set, block_size, batch_size, shuffle=False)

def generate_text(model, start_text, max_new_tokens=100, temperature=1.0, top_k=None):
    # encode start text to tokens
    idx = tf.constant([encode(start_text)], dtype=tf.int32)
    # generate
    idx = model.generate(idx, max_new_tokens=max_new_tokens,
                         temperature=temperature, top_k=top_k)
    # decode back to string
    return decode(idx.numpy()[0])

def main():
    model = Transformer()

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
    history = model.fit(train_ds,
                    validation_data=val_ds,
                    epochs=20)  # adjust epochs
    model.save("transformer_model.h5")  # HDF5 format
    generate_text(model, "The sea is")
    
 



main()