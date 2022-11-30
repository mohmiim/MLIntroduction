import json

from tensorflow.keras.applications import vgg19
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from tensorflow.keras.utils import pad_sequences, to_categorical
from tensorflow.keras.models import Model, load_model
from numpy import argmax

from pickle import load
import os


def extract_features(filename):
    # load the model
    model = vgg19.VGG19()
    # re-structure the model
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    # load the photo
    image = load_img(filename, target_size=(224, 224))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = vgg19.preprocess_input(image)
    # get features
    feature = model.predict(image, verbose=0)
    return feature


# map an integer to a word
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


# generate a description for an image
def generate_desc(model, tokenizer, photo, max_length):
    # seed the generation process
    in_text = 'startseq'
    # iterate over the whole length of the sequence
    for i in range(max_length):
        # integer encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad input
        sequence = pad_sequences([sequence], maxlen=max_length)
        # predict next word
        yhat = model.predict([photo, sequence], verbose=0)
        # convert probability to integer
        yhat = argmax(yhat)
        # map integer to word
        word = word_for_id(yhat, tokenizer)
        # stop if we cannot map the word
        if word is None:
            break
        # append as input for generating the next word
        in_text += ' ' + word
        # stop if we predict the end of the sequence
        if word == 'endseq':
            break
    return in_text


source_root = "D:/Training_Data/ImageCaption/"
output = os.path.join(source_root, "Flickr8k_Dataset", "converted")
tokenizer_file = os.path.join(output, "tokenizer.json")
max_length_file = os.path.join(output, "max_length.pkl")

# load the model
model = load_model('model-ep003-loss3.658-val_loss3.855.h5')
max_length = load(open(max_length_file, 'rb'))
tokenizer = Tokenizer()
with open(tokenizer_file) as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)

photo = extract_features('example2.jpg')
# generate description
description = generate_desc(model, tokenizer, photo, max_length)
description = description.split()[1:-1]
description = " ".join(description)
print(description)
