import tensorflow as tf

# import pdb

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
from transformer_code import *

import resources
import models

class CaptionGeneratorTransformer1():
    def __init__(self):
        logging.debug("Creating object for caption generation using transformer")
        #
        # self.vocab_size = 1652
        # self.embedding_dim = 200
        # self.max_length = 34
        #
        # self.pkl_wordtoix_file = resources.pkl_wordtoix_file
        # self.pkl_ixtoword_file = resources.pkl_ixtoword_file
        #
        # with open(self.pkl_wordtoix_file, "rb") as encoded_wordtoix_pickle:
        #     self.wordtoix = load(encoded_wordtoix_pickle)
        #
        # with open(self.pkl_ixtoword_file, "rb") as encoded_ixtoword_pickle:
        #     self.ixtoword = load(encoded_ixtoword_pickle)
        #
        # self.model_weights = models.model1_weights
        #
        # self.model_new = Model #Encoder
        # self.lstm_model = Model #Decoder
        #
        # self.HYP_ENABLE_ATTENTION = True


        # self.pkl_wordtoix_file = resources.pkl_wordtoix_file
        self.pkl_transformer_tokenizer_file = resources.pkl_transformer_tokenizer_file

        with open(self.pkl_transformer_tokenizer_file, "rb") as encoded_transformer_tokenizer_pickle:
            self.tokenizer = load(encoded_transformer_tokenizer_pickle)


        self.HYP_ENCODER_MODEL = "RESNET50"
        HYP_ENABLE_TRANSFORMER = True
        # self.image_features_extract_model=""
        # self.image_features_extract_model = tf.keras.Model
        HYP_ENCODER_MODEL = "RESNET50"
        HYP_ENABLE_TRANSFORMER = True
        if HYP_ENCODER_MODEL == "RESNET50":
            if HYP_ENABLE_TRANSFORMER:
                print("ResNet50 with Transformer")
                model = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3), pooling="avg")
                new_input = model.input
                hidden_layer = model.layers[-2].output
                self.image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

        # if HYP_ENABLE_TRANSFORMER:
        #     top_k = 5000
        #     # Building a Word embedding for top 5000 words in the captions
        #     tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,
        #                                                       oov_token="<unk>",
        #                                                       filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
        #     tokenizer.fit_on_texts(train_captions)
        #     train_seqs = self.tokenizer.texts_to_sequences(train_captions)


        logging.debug("SUCCESS: Creating object for caption generation using transformer")

    def encoder_model(self):
        logging.info("Setting up encoder model")
        HYP_ENCODER_MODEL = "RESNET50"
        HYP_ENABLE_TRANSFORMER = True
        if HYP_ENCODER_MODEL == "RESNET50":
            if HYP_ENABLE_TRANSFORMER:
                print("ResNet50 with Transformer")
                model = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3), pooling="avg")
                new_input = model.input
                hidden_layer = model.layers[-2].output
                self.image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

            # else:
            #     print("ResNet50 without Transformer")
            #     model = ResNet50(weights='imagenet')

            model_new = Model(model.input, model.layers[-2].output)

        logging.info("SUCCESS: Setting up encoder model")

    def model_architecture(self):
        logging.info("Setting up decoder model")

        self.transformer = tf.keras.models.load_model("D:\\git_repo\\aiml\\deploy_19aug_04\\src\\models\\save_image_caption_transformer_resnet50_tmp_dwn",
                                                      compile=False)
        import pdb
        pdb.set_trace()

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
        logging.info("SUCCESS: Setting up decoder model")

    # def update_model_weights(self):
    #     logging.info("Updating the model weight for decoder model")
    #     self.lstm_model.load_weights(self.model_weights)
    #     logging.info("SUCCESS: Updating the model weight for decoder model")

    def setup(self):
        logging.info("Setting up the model")
        # self.encoder_model()
        self.model_architecture()
        # self.update_model_weights()
        logging.info("COMPLETED: Setting up the model")

    '''
    """def preprocess(self, image_path):
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
        return x"""

    """def encode(self, image):
        """
        # Function to encode a given image into a vector of size (2048, )
        return: a 2048 vector of an image (Feature vector)
        """
        logging.info("Getting feature vector for the given image using encoder")
        image = self.preprocess(image)  # preprocess the image - user defined method
        fea_vec = self.model_new.predict(image)  # Get the encoding vector for the image
        fea_vec = np.reshape(fea_vec, fea_vec.shape[1])  # reshape from (1, 2048) to (2048, )
        # logging.debug("Feature vector for the image : ", fea_vec)
        return fea_vec"""

    """def greedySearch(self, photo_feature_vec):
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
        return final"""
    '''

    def load_image(self, image_path):
        import pdb
        pdb.set_trace()

        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (224, 224))
        return img, image_path

    def create_padding_mask(self, seq):
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
        return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

    def create_look_ahead_mask(self, size):
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask  # (seq_len, seq_len)

    def create_masks_decoder(self, tar):
        look_ahead_mask = self.create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = self.create_padding_mask(tar)
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
        return combined_mask



    def evaluate(self, image_file=""):
        """
        image_file = np.load(image_file.decode('utf-8') + '.npy')

        path_of_feature = p.numpy().decode("utf-8")
        np.save(path_of_feature, bf.numpy())

        p = image_file
        bf = img_tensor_val

        path_of_feature = image_file.numpy().decode("utf-8")
        np.save(path_of_feature, img_tensor_val.numpy())

        def map_func(img_name, cap):
            #get image feature vector
            img_tensor = np.load(img_name.decode('utf-8')+'.npy')
            return img_tensor, cap

        """
        #-----------------
        # path_of_feature = image_file.numpy().decode("utf-8")
        # np.save(path_of_feature, img_tensor_val.numpy())
        # img_tensor = np.load(self.load_image(image_file)[0].decode('utf-8') + '.npy')
        # img_tensor = np.load(self.load_image(image_file)[0].decode('utf-8') + '.npy')
        # img_tensor = np.load(image_file.decode('utf-8') + '.npy')

        # img_tensor_val = self.image_features_extract_model(img_tensor)


        # temp_input should be .npy

        #-----------------

        temp_input = tf.expand_dims(self.load_image(image_file)[0], 0)

        img_tensor_val = self.image_features_extract_model(temp_input)

        img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

        #pickle the tokenizer and use it here
        start_token = self.tokenizer.word_index['<start>']
        end_token = self.tokenizer.word_index['<end>']
        print("start_token", start_token)
        # decoder input is start token.
        decoder_input = [start_token]
        output = tf.expand_dims(decoder_input, 0)  # tokens
        print("output1", output.shape)
        result = []  # word list

        for i in range(100):
            dec_mask = self.create_masks_decoder(output)
            # print ("dec_mask", dec_mask.shape)
            # predictions.shape == (batch_size, seq_len, vocab_size)
            # predictions, attention_weights = transformer(img_tensor_val, output, False, dec_mask)

            predictions, attention_weights = self.transformer(img_tensor_val, output, False, dec_mask)
            # predictions, attention_weights = self.transformer_model(img_tensor_val, output, False, dec_mask)

            # select the last word from the seq_len dimension
            predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
            # return the result if the predicted_id is equal to the end token
            if predicted_id == end_token:
                return result, tf.squeeze(output, axis=0), attention_weights
            # concatentate the predicted_id to the output which is given to the decoder
            # as its input.
            result.append(self.tokenizer.index_word[int(predicted_id)])
            output = tf.concat([output, predicted_id], axis=-1)
        return result, tf.squeeze(output, axis=0), attention_weights

    def get_caption(self, input_image):
        logging.info("Generating a caption for given image")
        # image = encoding_test[image_name].reshape((1,2048))
        # image1 = self.encode(input_image).reshape((1, 2048))
        # x=plt.imread(images+pic)
        # plt.imshow(x)
        # plt.show()
        # caption = self.greedySearch(image1)

        caption_splitted, result, attention_weights = self.evaluate(image_file=input_image)
        caption = ' '.join(caption_splitted)

        print("Caption is generated: ", caption)
        return caption

    """def test_method(self, input_image=""):
        HYP_ENABLE_TRANSFORMER = True
        transformer = Model  # Decoder

        if HYP_ENABLE_TRANSFORMER:
            # top_k = vocab_size
            top_k = 8357

            num_layer = 4
            d_model = 512
            dff = 2048
            num_heads = 8
            row_size = 7  # I think this should match with output sixe of resnet 7X 7X 2048
            col_size = 7
            target_vocab_size = top_k + 1
            dropout_rate = 0.1

        # transformer = Transformer(num_layer, d_model, num_heads, dff, row_size, col_size, target_vocab_size,
        #                           max_pos_encoding=target_vocab_size, rate=dropout_rate)

        # self.model_weights = models.model1_weights
        print("--------")
        print(transformer)
        print("--------")
        import pdb
        pdb.set_trace()

        # transformer.save_weights(models.model1_weights)
        # transformer.load_weights("D:\\git_repo\\aiml\\deploy_19aug_04\\src\\models\\save_weights_image_caption_transformer_resnet50.h5")


        # input_image = "D:\\git_repoiml\\deploy_19aug_04\\input_files\\2090339522_d30d2436f9.jpg"

        #
        self.transformer = Transformer(num_layer, d_model, num_heads, dff, row_size, col_size, target_vocab_size, max_pos_encoding=target_vocab_size, rate=dropout_rate)


        ""
        # transformer()
        start_token = self.tokenizer.word_index['<start>']
        end_token = self.tokenizer.word_index['<end>']
        decoder_input = [start_token]
        output = tf.expand_dims(decoder_input, 0)  # token
        dec_mask = cap_tran_obj1.create_masks_decoder(output)
        test = tf.random.Generator.from_seed(123)
        test = test.normal(shape=(16, 49, 2048))
        self.transformer(test, output, False, dec_mask)
        # transformer.load_weights('model.h5')
        # ""
        self.transformer.load_weights(
            "D:\\git_repo\\aiml\\deploy_19aug_04\\src\\models\\save_weights_image_caption_transformer_resnet50.h5")
        caption_splitted, result, attention_weights = cap_tran_obj1.evaluate(image_file=input_image)
        print("----------------------")
        print(caption_splitted)
        print("----------------------")"""


if __name__ == "__main__":
    cap_tran_obj1 = CaptionGeneratorTransformer1()
    cap_tran_obj1.setup()

    start_token = cap_tran_obj1.tokenizer.word_index['<start>']
    end_token = cap_tran_obj1.tokenizer.word_index['<end>']

    # input_image = "D:\git_repo\aiml\deploy_19aug_04\input_files\\2090339522_d30d2436f9.jpg"
    input_image = "D:\\git_repo\\aiml\\deploy_19aug_04\\input_files\\2090339522_d30d2436f9.jpg"
    # cap_tran_obj1.test_method(input_image=input_image)
    caption_splitted, result, attention_weights = cap_tran_obj1.evaluate(image_file=input_image)
    # caption_splitted, result, attention_weights = cap_tran_obj1.test_method(input_image=input_image)
    # caption_splitted, result, attention_weights = cap_tran_obj1.evaluate(image_file=input_image)

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
        return super(Attention,self).get_config()"""

"""
if __name__ == "__main__":
    cap_gen = CaptionGenerator()
    cap_gen.setup()
    import pdb
    pdb.set_trace()
    cap_gen.get_caption(input_image="H:\\aiml\git_repo\\aiml\deploy_13aug_11\src\\2090339522_d30d2436f9.jpg")
"""




