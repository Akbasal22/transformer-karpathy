import tensorflow as tf
from tensorflow.keras.metrics import SparseCategoricalAccuracy
import numpy as np
import matplotlib.pyplot as plt
import random
from tokenizers import Tokenizer


# open input.txt
f = open('/content/drive/MyDrive/shakegpt/input.txt', 'r', encoding='utf-8')
text = f.read()
chars = sorted(list(set(text)))

#load tokenizer
tok = Tokenizer.from_file("/content/drive/MyDrive/shakegpt/bpe_tokenizer.json")


#global variables
block_size = 128
batch_size = 64
vocab_size = tok.get_vocab_size()
n_embed = 96
dropout = 0.2
num_heads = 6
n_layers = 6
learning_rate = 1e-4
head_size = n_embed//num_heads



##I will improve this with regex helped byte pair encoding,
def encode(text):
    return tok.encode(text).ids

def decode(ids):
    return tok.decode(ids)

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
        mask = tf.linalg.band_part(tf.ones((T, T)), -1, 0)  # (T, T)
        mask = tf.expand_dims(mask, 0)                      # (1, T, T)
        wei = tf.where(mask == 0, tf.fill(tf.shape(wei), -1e9), wei)
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
        out = self.proj(out)
        out = self.dropout(out)
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
        x = x + self.mlp(self.ln2(x))
        return x
    

class Transformer(tf.keras.Model):
    def __init__(self, vocab_size, n_embed, block_size, n_layers, num_heads, **kwargs):
        super().__init__(**kwargs)
        initializer = tf.keras.initializers.TruncatedNormal(stddev=0.02)
        #I should initalize these two in tensorflow layer type
        self.token_embedding_table = tf.keras.layers.Embedding(
            input_dim=vocab_size, output_dim=n_embed,
            embeddings_initializer=initializer
        )
        self.position_embedding_table = tf.keras.layers.Embedding(
            input_dim=block_size, output_dim=n_embed,
            embeddings_initializer=initializer
        )

        self.lm_head = tf.keras.layers.Dense(
            vocab_size, kernel_initializer=initializer
        )


        self.blocks = tf.keras.Sequential([Block(n_embed, num_heads) for _ in range(n_layers)])
        self.ln_f = tf.keras.layers.LayerNormalization()

        self.vocab_size = vocab_size
        self.n_embed = n_embed
        self.block_size = block_size
        self.n_layers = n_layers
        self.num_heads = num_heads
        
    def get_config(self):
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "n_embed": self.n_embed,
            "block_size": self.block_size,
            "n_layers": self.n_layers,
            "num_heads": self.num_heads,
        })
        return config
    def call(self, idx, targets = None):
        T = tf.shape(idx)[1]
        tok_emb = self.token_embedding_table(idx)
        positions = tf.range(start=0, limit=T, dtype=tf.int32) #???? explain
        pos_emb = self.position_embedding_table(positions)
        pos_emb = tf.expand_dims(pos_emb, axis=0)
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        return logits


    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=None):
        """
        idx: tensor (B, T) initial context
        returns: tensor (B, T + max_new_tokens) with generated tokens appended
        """
        for _ in range(max_new_tokens):
            # crop context to block_size
            idx_cond = idx[:, -block_size:] #take last block_size tokens
            logits = self.call(idx_cond)  # (B, T', vocab) run the model to get logits
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
    idx = tf.constant([encode(start_text)], dtype=tf.int32)
    idx = model.generate(idx, max_new_tokens=max_new_tokens,
                         temperature=temperature, top_k=top_k)
    return decode(idx.numpy()[0]).tolist()

def main():
    model = Transformer(vocab_size, n_embed, block_size, n_layers, num_heads)

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    
    model.compile(optimizer=optimizer,loss=loss_fn,metrics=[SparseCategoricalAccuracy()])
    history = model.fit(train_ds,
                    validation_data=val_ds,
                    epochs=10) 
    # model.save("model.h5")  # HDF5 format
    model.save("/content/drive/MyDrive/shakegpt/model.keras") #keras format  
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Validation Loss vs. Epoch")
    plt.legend()
    plt.show()
    generate_text(model, "The sea is")

my_model = tf.keras.models.load_model(
    "model.keras",
    custom_objects={
        "Transformer": Transformer,
        "Block": Block,
        "Head": Head,
        "MultiHeadAttention": MultiHeadAttention,
        "MLP": MLP
    },
)

generate_text(my_model, "Oh great")
main()