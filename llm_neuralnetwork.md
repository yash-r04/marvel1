# Neural Networks!
### what is neural networks ?
imagine keeping a high IQ and curious child but it only understands in math with infinite memory  in an environment full of active brain stimulating items, this is how i would describe neural networks

neural network is a subset of machine learning. neural network is an imitation of human brain where it trains itself to detect the underlying patterns with available data and predicts the outcome on unseen data.
**formal definition** : Neural Networks are computational models that mimic the complex functions of the human brain. The neural networks consist of interconnected nodes or neurons that process and learn from data, enabling tasks such as pattern recognition and decision making in machine learning.


in the simplest way possible this is how neural networks can be described - 
> input computation, output generation, and iterative refinement enhancing the network’s proficiency in diverse tasks. 

### parts of neural network
[![Screenshot-2024-11-02-232734.png](https://i.postimg.cc/t4kDSJTK/Screenshot-2024-11-02-232734.png)](https://postimg.cc/NKy1LgRD)

- input layer : recieves data input and passes it to the hidden layers.
- hidden layers : sandwitched between the input and output layer, this is the comupational zone of neural networks
- output layer : it gives the output as a probablity and sends feedback to the hidden layers so that the wieghts and bias will change.
  
neurons are the core processing unit of neural network and layers of neurons form a neural network.

neurons of one layer is connected to another layer of neurons through channels and each channel a weight (assigned numerical value)

input is multipled to corresponding weights and the sum of it is sent as input to hidden layer 

each neuron in the hidden layer has a bias (assoiciated with a numerical value) and bias is added to the input and passed on to a threshold function called activation function, based on it’s result



1. **Artificial Neural Network (ANN)**
    - Definition: A group of perceptrons or neurons layered in a feed-forward manner.
    - Characteristics:
        - Processes data in one direction.
        - May or may not have hidden layers.
      Advantages:
        - Stores information across the network.
        - Works with incomplete knowledge.
        - Offers fault tolerance and distributed memory.
      Disadvantages:
        -  It is highly dependent on the hardware
       
        - Requires trial and error for optimal structure. appropiate network structure is found through trial and error
2. **Convolutional Neural Network (CNN)**
    - Definition: A model using multilayer perceptrons with convolutional layers .
    - Characteristics:
        - Creates feature maps for image regions.
    - **Advantages:**
        - High accuracy in image recognition.
        - Automatically detects important features.
        - Implements weight sharing.
    - **Disadvantages:**
        - Does not encode object position or orientation.
        - Spatial invariance issues with input data.
        - Requires substantial training data.
3. **Recurrent Neural Network (RNN) - self learn!!**
    - Definition: Complex networks that save and reuse output.
    - Characteristics:
        - Learning from feedback, promotes continuous computation.
    - **Advantages:**
        - Remembers information over time via Long Short-Term Memory (LSTM).
        - Compatible with convolutional layers for enhancing pixel neighborhoods.
    - **Disadvantages:**
        - Prone to gradient vanishing/exploding.
        - Challenging to train.
        - Limited processing of long sequences.
4. **Comparison Summary**
    - **Data Types:**
        - ANN: Tabular/Text data.
        - CNN: Image data.
        - RNN: Sequence data.
    - **Parameter Sharing:**
        - Not possible in ANN; possible in CNN and RNN.
    - **Input Length:**
        - Fixed length for ANN and CNN; variable for RNN.
    - **Recurrent Connections:**
        - Not possible in ANN and CNN; achievable in RNN.
    - **Spatial Relationships:**
        - Available only in CNN.
    - **Power:**
        - ANN < CNN, RNN; CNN > ANN, RNN.
    - **Performance:**
        - RNN has less feature compatibility compared to CNN.
### types of neural networks

#### ANN
#### CNN
#### RNN


### use cases in real life
1. natural language processing
2. speech recognition
3. image proccessing
4. autonomous vehicles!
   and more

# LARGE language models

# Resources used to write this article
[statquest](https://www.youtube.com/watch?v=CqOfi41LfDw&t=134s)

[Neural Network Architectures & Deep Learning](https://www.youtube.com/watch?v=oJNHXPs0XDk)

completed this as a side quest!
[![Screenshot-2024-11-02-233445.png](https://i.postimg.cc/85jpY7bs/Screenshot-2024-11-02-233445.png)](https://postimg.cc/7GrrGYdy)
