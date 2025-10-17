import sys
import tensorflow as tf

from train import generate_text, Transformer

my_model = tf.keras.models.load_model(
    '/content/drive/MyDrive/shakegpt/model.keras',
    custom_objects={'Transformer': Transformer},
)

text = generate_text(my_model, "tell me a joke", 100)
#this is a helper function to nicely format transformer's output
def remove_space_and_skip(s):
    s = list(s)
    result = []
    i = 0
    while i < len(s)-1:
        if s[i] == " ":
            result.append(s[i+1])
            i += 2
        else:
            result.append(s[i])
            i += 1
    yeni = "".join(result)
    print(yeni)

remove_space_and_skip(text)

#example output

"""

tell me a jokewed.

GRUMIO:
My lord, if with a counselved rich the rest
When do to swisdom our needed this to perjourn.

MARIANA:
My lord, I will go;
The good, I heart as thou father for twen


"""