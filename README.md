# AIML - Project
# Automatic Image Captioning

### What is Image Captioning?
Image Captioning is the process of generating textual description of an image. It uses both Natural Language Processing and Computer Vision to generate the captions. Image Caption Generator or Photo Descriptions is one of the Applications of Deep Learning.

### How does it work?
In Which we have to pass the image to the model and the model does some processing and generating captions or descriptions as per its training. 

### Does it work perfectly?
Yes, It works well if we train the model on a large dataset and high performance machine. However, the described model's prediction is sometimes not that much accurate and generates some meaningless sentences. We need very high computational power and a very huge dataset for better results.

### Why captioning?
Captioning the images with proper descriptions automatically has become an interesting and challenging problem. Some of the applications are listed below,
- SkinVision : Lets you confirm weather a skin condition can be skin cancer or not.
- Google Photos: Classify your photo into Mountains, sea etc.
- Deepmind: Achieved superhuman level playing Game Atari.
- Facebook: Using AI to classify, segmentate and finding patterns in pictures.
- A U.S. company is predicting crop yield using images from satellite.
- Fed Ex and other courier services: Are using hand written digit recognition system from may times now to detect pin code correctly.
- Picasa : Using facial Recognition to identify your friends and you in a group picture.
- Tesla/Google Self Drive Cars: All the self drive cars are using image/video processing with neural network to attain their goal.

### What is the architecture used?
#### We have used multiple architectures to generate models as mentioned below,
- ResNet50+LSTM
- ResNet50+LSTM+Optimizer
- ResNet50+LSTM+Attention
- ResNet50+LSTM+Attention+Optimizer
- ResNet50+Transformer

We used one joint model AICRL, which is able to conduct the automatic image captioning based on ResNet50 and LSTM with soft attention. AICRL consists of one encoder and one decoder. 

**Encoder**

The Convolutional Neural Network(CNN) can be thought of as an encoder. The input image is given to CNN to extract the features. The last hidden state of the CNN is connected to the Decoder. The encoder adopts ResNet50 based on the convolutional neural network, which creates an extensive representation of the given image by embedding it into a fixed length vector. 

**Decoder**

The Decoder is a Recurrent Neural Network(RNN) which does language modelling up to the word level. The decoder is designed with LSTM, a recurrent neural network and a soft attention mechanism, to selectively focus the attention over certain parts of an image to predict the next sentence. 

![image](https://user-images.githubusercontent.com/42552004/185698360-a42be897-b6e3-4e8a-bf8b-160b4b0a4599.png)

We used **Transformers model** to get a better results,

The transformer network employs an encoder-decoder architecture similar to that of an RNN. The main difference is that transformers can receive the input sentence/sequence in parallel, i.e, there is no time step associated with the input, and all the words in the sentence can be passed simultaneously.

![image](https://user-images.githubusercontent.com/42552004/189497685-81cf89da-2592-4e59-b879-555adf828b49.png)


### What are the Pre-requisites?
This project requires good knowledge of Deep learning, Python, working on Jupyter/Colab notebooks, Keras library, Numpy, and Natural language Processing. Make sure you have installed all the following necessary libraries:
- Tensorflow
- Keras
- Pandas
- NumPy
- nltk ( Natural language tool kit)

### What data sets are used for training the model?
- Flickr 8K. A collection of 8 thousand described images taken from flickr.com.
- Flickr 30K. A collection of 30 thousand described images taken from flickr.com.
- Common Objects in Context (COCO). A collection of more than 120 thousand images with descriptions

### What approaches we have taken for finding a best model?
We have trained multiple pre-trained models with various hyper parameters and finalise the model which is giving a good accuracy as described the below,
<table>
<thead>
  <th>Dataset</th>
  <th>Encoder Model</th>
  <th>WordEmbedding</th>
  <th>Attention</th>
  <th>Transformer</th>
  <th>Loss function</th>
  <th>Optimizer</th>
  <th>No of pics per batch</th>
  <th>Learning rate</th>
  <th>Epochs</th>
</thead>
<tbody>
<tr>
  <th>Flicker8k, Flicker30k, COCO123k</th>
  <th>VGG16, RESNET50, INCEPTIONV3</th>
  <th>GLOVE, word2vec</th>
  <th>With/Without Attention</th>
  <th>With/Without Transformer</th>
  <th>categorical_crossentropy</th>
  <th>adam</th>
  <th>30</th>
  <th>0.0001</th>
  <th>5, 10, 30</th>
<tbody>
</tr>
</table>


### Can I train a model?
Yes, You can train the model using **the Google Colab notebook with various parameters**
[Google_colab_notebook_automatic_image_captioning](https://github.com/chirucodes/aiml/blob/main/automatic_image_captioning/ResNet50_LSTM_with_Attention_ImageCaptioningGroup4.ipynb)

### Some observations during the model training?
You can find the observation in the [google sheets](https://docs.google.com/spreadsheets/d/1eO9BrHyBLZACl_1QBHOROR3yUCW5jHyael8NUIjX4vA/edit#gid=0)

### What all should I deploy on a remote server?
https://github.com/chirucodes/aiml/tree/main/deployment_content <br><br>
![image](https://user-images.githubusercontent.com/42552004/185702837-1a550f5d-d1d8-4dd4-8de1-f65d1f2e3bbb.png)


### Is it live?
Yes, you can access the hosted service using the below links,
- http://aicg4live.westus3.cloudapp.azure.com/
- https://aiml-phase1.azurewebsites.net/
- https://aicg4.deta.dev/docs
