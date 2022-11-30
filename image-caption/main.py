import io
import json

import tensorflow as tf
from numpy import array
from tensorflow.keras.applications import vgg19, vgg16, xception, convnext
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from tensorflow.keras.utils import pad_sequences, to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint

from pickle import dump, load
from enum import Enum
import os
import string

from tensorflow.keras import Input
from tensorflow.keras.layers import Dropout, Dense, Embedding, LSTM, Add


class Extractor(Enum):
    CON_NEXT = 1
    VGG_19 = 2
    VGG_16 = 3
    XCEPTION = 4


START_TOKEN = "startseq"
END_TOKEN = "endseq"
PREP_DATA = False
PREP_TRAIN = False
TRAIN = True
active_extractor = Extractor.VGG_19
Image_Width = 224
Image_Height = 224
Image_Depth = 3
target_size = (Image_Width, Image_Height)
targetSize_withdepth = (Image_Width, Image_Height, Image_Depth)
epochs = 200

print("START")
source_root = "D:/Training_Data/ImageCaption/"
source = os.path.join(source_root, "Flickr8k_Dataset", "data")
output = os.path.join(source_root, "Flickr8k_Dataset", "converted")
text_root = os.path.join(source_root, "Flickr8k_text")
text_source = os.path.join(text_root, "Flickr8k.token.txt")
X1_file = os.path.join(output, "X1.pkl")
X2_file = os.path.join(output, "X2.pkl")
y_file = os.path.join(output, "y.pkl")
X1_Dev_file = os.path.join(output, "X1_dev.pkl")
X2_Dev_file = os.path.join(output, "X2_dev.pkl")
y_Dev_file = os.path.join(output, "y_dev.pkl")
tokenizer_file = os.path.join(output, "tokenizer.json")
max_length_file = os.path.join(output, "max_length.pkl")
voc_size_file = os.path.join(output, "voc_size.pkl")

print(f"Tensorflow:  {tf.__version__}");


def get_feature_extraction_fn():
    match active_extractor:
        case Extractor.CON_NEXT:
            return convnext.ConvNeXtXLarge, convnext.preprocess_input
        case Extractor.VGG_19:
            return vgg19.VGG19, vgg19.preprocess_input
        case Extractor.VGG_16:
            return vgg16.VGG16, vgg16.preprocess_input
        case Extractor.XCEPTION:
            return xception.Xception, xception.preprocess_input


def extract_features(fe, proc_fn, source_dir, dest):
    fe.summary()
    features = dict()
    count = 0
    for input_name in os.listdir(source_dir):
        fullname = os.path.join(source_dir, input_name)
        image = load_img(fullname, target_size=target_size)
        image = img_to_array(image)
        # predict needs array of images so we add extra dimension
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        # prep the image
        image = proc_fn(image)
        extracted_features = fe.predict(image, verbose=0)
        image_name = input_name.split('.')[0]
        features[image_name] = extracted_features
        print(f">{image_name} No {count}")
        count += 1
    output_file = os.path.join(dest, "features.pkl")
    dump(features, open(output_file, 'wb'))
    print(count)


def load_text_from_file(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text


def convert_str_to_id_and_description(str_data):
    id_to_description = dict()
    # prepare translation table for removing punctuation
    table = str.maketrans('', '', string.punctuation)
    # build a list of all description strings
    vocabulary = set()
    # process lines
    for line in str_data.split('\n'):
        tokens = line.split()
        if len(line) < 2:
            continue
        # take the first token as the image id, the rest as the description
        image_id, image_desc = tokens[0], tokens[1:]
        # remove filename from image id
        image_id = image_id.split('.')[0]
        image_desc = clean_description(image_desc, table)
        vocabulary.update(image_desc)
        # convert description tokens back to string
        image_desc = ' '.join(image_desc)
        # create the list if needed
        if image_id not in id_to_description:
            id_to_description[image_id] = list()
        # store description
        id_to_description[image_id].append(image_desc)
    return id_to_description, vocabulary


def clean_description(description, table):
    cleaned_desc = [word.lower() for word in description]
    # remove punctuation from each token
    cleaned_desc = [w.translate(table) for w in cleaned_desc]
    # remove hanging 's' and 'a'
    cleaned_desc = [word for word in cleaned_desc if len(word) > 1]
    # remove tokens with numbers in them
    cleaned_desc = [word for word in cleaned_desc if word.isalpha()]
    # store as string
    return cleaned_desc


def save_converted_descriptions(data, output_dir):
    filename = os.path.join(output_dir, "descriptions.txt")
    lines = list()
    for key, desc_list in data.items():
        for desc in desc_list:
            lines.append(key + ' ' + desc)
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()


def load_data_set_ids(data_set_file):
    data_str = load_text_from_file(data_set_file)
    data_set = list()
    for entry in data_str.split("\n"):
        if len(entry) > 0:
            image_id = entry.split(".")[0]
            data_set.append(image_id)
    return set(data_set)


def load_cleaned_descriptions(file_name, ids_to_use):
    data_str = load_text_from_file(file_name)
    text_descriptions = dict()
    max_length = 0
    for entry in data_str.split("\n"):
        tokens = entry.split()
        image_id = tokens[0]
        image_desc = tokens[1:]
        if image_id in ids_to_use:
            if image_id not in text_descriptions:
                text_descriptions[image_id] = list()
            desc = START_TOKEN + " " + ' '.join(image_desc) + " " + END_TOKEN
            length = len(image_desc) + 2
            if max_length < length:
                max_length = length
            text_descriptions[image_id].append(desc)
    return text_descriptions, max_length


def load_image_features(file_name, ids_to_use):
    all_images_features = load(open(file_name, 'rb'))
    images_features = {image: all_images_features[image] for image in ids_to_use}
    return images_features


if PREP_DATA:
    fe_fn, proc_fn = get_feature_extraction_fn()
    feature_extractor = fe_fn()
    feature_extractor = Model(inputs=feature_extractor.inputs, outputs=feature_extractor.layers[-2].output)
    extract_features(feature_extractor, proc_fn, source, output)
    # load descriptions
    str_contents = load_text_from_file(text_source)
    # parse descriptions
    description_data, voc = convert_str_to_id_and_description(str_contents)
    print(f"Loaded: {len(description_data)}")
    print(f"Vocabulary Size: {len(voc)}")
    save_converted_descriptions(description_data, output)
    print("DONE")


def get_tokenizer(data):
    desc = list()
    for key in data.keys():
        [desc.append(d) for d in data[key]]
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(desc)
    return tokenizer


def prep_sequence(tokenizer, max_length, description_data, image_data, voc_size):
    X1, X2, y = list(), list(), list()
    for key, desc_list in description_data.items():
        for desc in desc_list:
            seq = tokenizer.texts_to_sequences([desc])[0]
            for i in range(1, len(seq)):
                in_seq, out_seq = seq[:i], seq[i]
                in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                out_seq = to_categorical([out_seq], num_classes=voc_size)[0]
                X1.append(image_data[key][0])
                X2.append(in_seq)
                y.append(out_seq)
    return array(X1), array(X2), array(y)


source_root = "D:/Training_Data/ImageCaption/"
source = os.path.join(source_root, "Flickr8k_Dataset", "data")
output = os.path.join(source_root, "Flickr8k_Dataset", "converted")
text_source = os.path.join(source_root, "Flickr8k_text", "Flickr8k.token.txt")

if PREP_TRAIN:
    training_file = os.path.join(text_root, "Flickr_8k.trainImages.txt")
    descriptions_file = filename = os.path.join(output, "descriptions.txt")
    feature_file = os.path.join(output, "features.pkl")
    train_ids = load_data_set_ids(training_file)
    print(f"Training Samples IDs: {len(train_ids)}")
    training_desc, max_desc_length = load_cleaned_descriptions(descriptions_file, train_ids)
    print(f"Training Descriptions Max Length: {max_desc_length}")
    print(f"Training Description count : {len(training_desc)}")
    training_image_features = load_image_features(feature_file, train_ids)
    print(f"Training Images features: {len(training_image_features)}")
    tokenizer = get_tokenizer(training_desc)

    voc_size = len(tokenizer.word_index) + 1
    print(f"Voc size = {voc_size}")
    X1, X2, y = prep_sequence(tokenizer, max_desc_length, training_desc, training_image_features, voc_size)
    print(f"X1 > {X1.shape}")
    print(f"X2 > {X2.shape}")
    print(f"y > {y.shape}")
    tokenizer_json = tokenizer.to_json()
    with io.open(tokenizer_file, 'w', encoding='utf-8') as f:
        f.write(json.dumps(tokenizer_json, ensure_ascii=False))
    dump(X1, open(X1_file, 'wb'))
    dump(X2, open(X2_file, 'wb'))
    dump(y, open(y_file, 'wb'))
    dump(voc_size, open(voc_size_file, 'wb'))
    dump(max_desc_length, open(max_length_file, 'wb'))

    dev_file = os.path.join(text_root, "Flickr_8k.devImages.txt")
    dev_ids = load_data_set_ids(dev_file)
    print(f"Training Samples IDs: {len(dev_ids)}")
    dev_desc, dev_max_desc_length = load_cleaned_descriptions(descriptions_file, dev_ids)
    print(f"Training Descriptions Max Length: {dev_max_desc_length}")
    print(f"Training Description count : {len(dev_desc)}")
    dev_image_features = load_image_features(feature_file, dev_ids)
    print(f"Training Images features: {len(dev_image_features)}")

    X1_Dev, X2_Dev, y_dev = prep_sequence(tokenizer, max_desc_length, dev_desc, dev_image_features, voc_size)
    print(f"X1_Dev > {X1_Dev.shape}")
    print(f"X2_Dev > {X2_Dev.shape}")
    print(f"y_Dev > {y_dev.shape}")
    dump(X1_Dev, open(X1_Dev_file, 'wb'))
    dump(X2_Dev, open(X2_Dev_file, 'wb'))
    dump(y_dev, open(y_Dev_file, 'wb'))


# define the captioning model
def define_model(vocab_size, max_length):
    # feature extractor model
    inputs1 = Input(shape=(4096,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    # sequence model
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    # decoder model
    decoder1 = Add()([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    # tie it together [image, seq] [word]
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    # summarize model
    print(model.summary())
    return model


if TRAIN:
    max_length = load(open(max_length_file, 'rb'))
    voc_size = load(open(voc_size_file, 'rb'))
    tokenizer = Tokenizer()
    with open(tokenizer_file) as f:
        data = json.load(f)
        tokenizer = tokenizer_from_json(data)
    print(f"Max Length {max_length}")
    print(f"Voc Size {voc_size}")
    print(f"Voc size (Tok) = {len(tokenizer.word_index) + 1}")

    X1_Dev = load(open("c:/data/X1_Dev.pkl", 'rb'))[:1000]
    X2_Dev = load(open("c:/data/X2_Dev.pkl", 'rb'))[:1000]
    y_dev = load(open("c:/data/y_dev.pkl", 'rb'))[:1000]

    print(f"X1_Dev > {X1_Dev.shape}")
    print(f"X2_Dev > {X2_Dev.shape}")
    print(f"y_Dev > {y_dev.shape}")

    X1 = load(open("c:/data/X1.pkl", 'rb'))
    X2 = load(open("c:/data/X2.pkl", 'rb'))
    y = load(open("c:/data/y.pkl", 'rb'))

    print(f"X1 > {X1.shape}")
    print(f"X2 > {X2.shape}")
    print(f"y > {y.shape}")

    # define the model
    model = define_model(voc_size, max_length)
    # define checkpoint callback
    filepath = 'model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', save_best_only=False, mode='min')

    # fit model
    model.fit([X1, X2], y, epochs=50, callbacks=[checkpoint], validation_data=([X1_Dev, X2_Dev], y_dev))
