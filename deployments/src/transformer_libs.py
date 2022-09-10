# Import modules
import logging
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications.resnet50 import ResNet50

# LSTM architecture
from keras import Input
from keras.layers import Dropout, Dense, Embedding, LSTM
from keras.layers import add
from keras.layers import Layer
import keras.backend as K
from keras.models import Model

from keras_preprocessing.sequence import pad_sequences

from keras.preprocessing import image
import numpy as np
from pickle import dump, load
import matplotlib.pyplot as plt

import tensorflow.keras.utils as tf_utils


import resources
import models

class CaptionGeneratorTransformer:
    def __init__(self):
        logging.debug("Creating object for caption generation")

        # self.vocab_size = 1652
        # self.embedding_dim = 200
        # self.max_length = 34

        # self.pkl_wordtoix_file = resources.pkl_wordtoix_file
        # self.pkl_ixtoword_file = resources.pkl_ixtoword_file

        # with open(self.pkl_wordtoix_file, "rb") as encoded_wordtoix_pickle:
        #     self.wordtoix = load(encoded_wordtoix_pickle)

        # with open(self.pkl_ixtoword_file, "rb") as encoded_ixtoword_pickle:
        #     self.ixtoword = load(encoded_ixtoword_pickle)

        self.model_weights = models.transformer_model1_weights

        self.model_new = Model #Encoder
        # self.lstm_model = Model #Decoder

        # self.HYP_ENABLE_ATTENTION = True
        # """
        self.HYP_ENCODER_MODEL = "RESNET50"

        logging.debug("SUCCESS: Creating object for caption generation")


    def encoder_model(self):
        logging.info("Setting up encoder model")
        if self.HYP_ENCODER_MODEL == "RESNET50":
            # model = ResNet50(weights='imagenet')
            # self.model_new = Model(model.input, model.layers[-2].output)
            # Create a new model, by removing the last layer (output layer) from the ResNet50
            print("ResNet50 with Transformer")
            model = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3), pooling="avg")
            new_input = model.input
            hidden_layer = model.layers[-2].output
            image_features_extract_model = tf.keras.Model(new_input, hidden_layer)
        logging.info("SUCCESS: Setting up encoder model")

    def model_architecture(self):
        logging.info("Setting up decoder model")
        """
        if self.HYP_ENABLE_ATTENTION:
            inputs1 = Input(shape=(2048,))  # 2048 is the output of the RESNET50
            fe1 = Dropout(0.5)(inputs1)  # Dropping off 50% from the input - 1024
            fe2 = Dense(256, activation='relu')(fe1)  # relu 0 if <0.5 and 1 if >=0.5

            inputs2 = Input(shape=(34,))
            # inputs=Input((features,))
            # embedding_dim - replace at output_dim
            # x=Embedding(input_dim=vocab_size, output_dim=200, input_length=features, embeddings_regularizer=keras.regularizers.l2(.001))(inputs1)
            x = Embedding(self.vocab_size, self.embedding_dim, mask_zero=True)(inputs2)  # embedding_dim = 200, vocab_size = 8762

            # se3 = LSTM(256)(se2)
            att_in = LSTM(256, return_sequences=True, dropout=0.3, recurrent_dropout=0.2)(x)

            att_out = Attention()(att_in)

            decoder1 = add([fe2, att_out])

            decoder2 = Dense(256, activation='relu')(decoder1)

            outputs = Dense(self.vocab_size, activation='softmax')(decoder2)

            self.lstm_model = Model(inputs=[inputs1, inputs2], outputs=outputs)
        """
        """if HYP_ENABLE_TRANSFORMER:
            # top_k = vocab_size

            num_layer = 4
            d_model = 512
            dff = 2048
            num_heads = 8
            row_size = 7  # I think this should match with output sixe of resnet 7X 7X 2048
            col_size = 7
            target_vocab_size = top_k + 1
            dropout_rate = 0.1
        """

        logging.info("SUCCESS: Setting up decoder model")

    def update_model_weights(self):
        logging.info("Updating the model weight for decoder model")
        self.lstm_model.load_weights(self.model_weights)
        logging.info("SUCCESS: Updating the model weight for decoder model")

    def setup(self):
        logging.info("Setting up the model")
        self.encoder_model()
        self.model_architecture()
        self.update_model_weights()
        logging.info("COMPLETED: Setting up the model")


    def preprocess(self, image_path):
        # Convert all the images to size 299x299 as expected by the inception v3 model(To be replaced with ResNet50)
        # from keras.preprocessing import image

        # img = image.load_img(image_path, target_size=(299, 299))
        # img = image.load_img(image_path, target_size=(224, 224))
        img = tf_utils.load_img(image_path, target_size=(224, 224))
        # Convert PIL image to numpy array of 3-dimensions
        # x = image.img_to_array(img)
        x = tf_utils.img_to_array(img)
        # Add one more dimension
        x = np.expand_dims(x, axis=0)
        # preprocess the images using preprocess_input() from inception module
        # from keras.applications.inception_v3 import preprocess_input
        # from keras.applications.resnet50 import preprocess_input
        x = preprocess_input(x)
        return x

    def encode(self, image):
        """
        # Function to encode a given image into a vector of size (2048, )
        return: a 2048 vector of an image (Feature vector)
        """
        logging.info("Getting feature vector for the given image using encoder")
        image = self.preprocess(image)  # preprocess the image - user defined method
        fea_vec = self.model_new.predict(image)  # Get the encoding vector for the image
        fea_vec = np.reshape(fea_vec, fea_vec.shape[1])  # reshape from (1, 2048) to (2048, )
        # logging.debug("Feature vector for the image : ", fea_vec)
        return fea_vec

    def greedySearch(self, photo_feature_vec):
        logging.info("Preparing the caption for the image")
        in_text = 'startseq'
        for i in range(self.max_length):
            sequence = [self.wordtoix[w] for w in in_text.split() if w in self.wordtoix]
            sequence = pad_sequences([sequence], maxlen=self.max_length)
            yhat = self.lstm_model.predict([photo_feature_vec, sequence], verbose=0)
            yhat = np.argmax(yhat)
            word = self.ixtoword[yhat]
            in_text += ' ' + word
            if word == 'endseq':
                break
        final = in_text.split()
        final = final[1:-1]
        final = ' '.join(final)
        return final

    def get_caption(self, input_image):
        logging.info("Generating a caption for given image")
        # image = encoding_test[image_name].reshape((1,2048))
        image1 = self.encode(input_image).reshape((1, 2048))
        # x=plt.imread(images+pic)
        # plt.imshow(x)
        # plt.show()
        caption = self.greedySearch(image1)
        print("Caption is generated: ", caption)
        return caption

"""
class Attention(Layer):
    def __init__(self,**kwargs):
        super(Attention,self).__init__(**kwargs)

    def build(self,input_shape):
        self.W=self.add_weight(name="att_weight",shape=(input_shape[-1],1),initializer="normal")
        self.b=self.add_weight(name="att_bias",shape=(input_shape[1],1),initializer="zeros")
        super(Attention, self).build(input_shape)

    def call(self,x):
        et=K.squeeze(K.tanh(K.dot(x,self.W)+self.b),axis=-1)
        at=K.softmax(et)
        at=K.expand_dims(at,axis=-1)
        output=x*at
        return K.sum(output,axis=1)

    def compute_output_shape(self,input_shape):
        return (input_shape[0],input_shape[-1])

    def get_config(self):
        return super(Attention,self).get_config()

"""

"""
if __name__ == "__main__":
    cap_gen = CaptionGenerator()
    cap_gen.setup()
    import pdb
    pdb.set_trace()
    cap_gen.get_caption(input_image="H:\\aiml\git_repo\\aiml\deploy_13aug_11\src\\2090339522_d30d2436f9.jpg")
"""
