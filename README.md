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

Now we will see some information about the dataset and the architecture of the neural network of the Image captions generator.

![image](https://user-images.githubusercontent.com/42552004/185695247-312e4c1f-f893-4869-ae7a-1088ae27661e.png)




### Pre-requisites:
This project requires good knowledge of Deep learning, Python, working on Jupyter notebooks, Keras library, Numpy, and Natural language Processing

Make sure you have installed all the following necessary libraries:

- Tensorflow
- Keras
- Pandas
- NumPy
- nltk ( Natural language tool kit)

Encoder
The Convolutional Neural Network(CNN) can be thought of as an encoder. The input image is given to CNN to extract the features. The last hidden state of the CNN is connected to the Decoder.

Decoder
The Decoder is a Recurrent Neural Network(RNN) which does language modelling up to the word level. The first time step receives the encoded output from the encoder and also the <START> vector.
  

### Colab file to execute
[automatic_image_captioning/ResNet50_LSTM_with_Attention_ImageCaptioningGroup4.ipynb](https://github.com/chirucodes/aiml/blob/main/automatic_image_captioning/ResNet50_LSTM_with_Attention_ImageCaptioningGroup4.ipynb)


### Executions_n_observations:
https://docs.google.com/spreadsheets/d/1eO9BrHyBLZACl_1QBHOROR3yUCW5jHyael8NUIjX4vA/edit#gid=0


### Source of Deployments to deta.sh
https://github.com/chirucodes/aiml/tree/main/first_micro

### Access Url
https://aicg4.deta.dev/docs
