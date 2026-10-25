#  Emojify (NLP - Word Vectors & LSTM Model)

This project implements an NLP model (Emojify-v2) that takes a sentence as input and automatically predicts the most appropriate emoji using pre-trained GloVe word embeddings and deep LSTM networks built with Keras and TensorFlow.

---

##  Core Components & Functions

The project is structured around several key modules and neural network layers:

### 1. `sentence_to_avg`
* **Description:** Converts an input sentence into a single 50-dimensional vector by taking the element-wise average of the GloVe vector representations of its words.

### 2. `model` (NumPy Softmax Classifier)
* **Description:** Implements a custom neural network training loop in NumPy utilizing GloVe averaged vectors, forward propagation, cross-entropy loss, and stochastic gradient descent (SGD).

### 3. `sentences_to_indices`
* **Description:** Converts an array of sentences into sequences of numerical word indices based on a vocabulary mapping, applying zero-padding to ensure uniform length for the embedding layer.

### 4. `pretrained_embedding_layer`
* **Description:** Creates and loads a non-trainable Keras `Embedding` layer initialized with pre-trained GloVe 50-dimensional vectors, adding an extra row for unknown words (`<UNK>`).

### 5. `Emojify_V2` (Keras LSTM Architecture)
* **Description:** Builds the deep sequence model graph containing:
  * Input layer for token indices.
  * Pre-trained embedding layer.
  * Two sequential `LSTM` layers (with 128 hidden units each, incorporating dropout for regularization).
  * A dense output layer with a softmax activation to predict across 5 emotion/emoji classes.

---
