# Neural Networks!
![image](https://github.com/user-attachments/assets/4de1d162-a678-49fc-9eb0-f7200401d953)
### what is neural networks ?
imagine keeping a high IQ and curious child but it only understands in math with infinite memory  in an environment full of active brain stimulating items, this is how i would describe neural networks

neural network is a subset of machine learning. neural network is an imitation of human brain where it trains itself to detect the underlying patterns with available data and predicts the outcome on unseen data.
**formal definition** : Neural Networks are computational models that mimic the complex functions of the human brain. The neural networks consist of interconnected nodes or neurons that process and learn from data, enabling tasks such as pattern recognition and decision making in machine learning.


in the simplest way possible this is how neural networks can be described - 
> input computation, output generation, and iterative refinement enhancing the network’s proficiency in diverse tasks. 

### parts of neural network
#### neurons or nodes:
It is the core processing unit of neural network and layers of neurons form a neural network. It Receive input, process it, and pass the output to the next layer.

neurons of one layer is connected to another layer of neurons through channels and each channel a weight (assigned numerical value)
[![Screenshot-2024-11-02-232734.png](https://i.postimg.cc/t4kDSJTK/Screenshot-2024-11-02-232734.png)](https://postimg.cc/NKy1LgRD)

- input layer : recieves data input and passes it to the hidden layers.
- hidden layers : sandwitched between the input and output layer, this is the comupational zone of neural networks
- output layer : it gives the output as a probablity and sends feedback to the hidden layers so that the wieghts and bias will change.

#### Weights
It is the Parameters that are adjusted during training and essentially represents the strength of the connection between neurons.
#### Bias
An additional parameter in each neuron that allows the model to fit the data better.

####  Activation Function

It is a mathematical function is applied to the output of each neuron and used to introduce non-linearity to the model.
Common activation functions include:
- Sigmoid
  
   $\ \sigma(x) = \frac{1}{1+e^{(-x)}} \$
- Tanh
  $f(x) = \frac{(e^x - e^-x) }{ (e^x + e^-x)}$
- ReLU (Rectified Linear Unit) - widely used
  $f(x) = max{0, z} $
- Softmax
  
![image](https://www.gstatic.com/education/formulas2/553212783/en/softmax_function.svg)
### working of neural networks
![image](https://github.com/user-attachments/assets/9793d393-3e16-4363-8979-b96fb2dcf3b9)

The learning (training) process of a neural network is an iterative process in which the calculations are carried out forward and backward through each layer in the network until the loss function is minimized.

The entire learning process can be divided into three main parts:

  1. Forward propagation (Forward pass)
  2. Calculation of the loss function
  3. Backward propagation (Backward pass/Backpropagation)
    
input is multipled to corresponding weights and the sum of it is sent as input to hidden layer 

each neuron in the hidden layer has a bias (assoiciated with a numerical value) and bias is added to the input and passed on to a threshold function called activation function, based on it’s result

1.Forward Propagation

Input data is fed into the input layer.
Data is passed through hidden layers, where neurons process it using weights, biases, and activation functions.
Final output is produced at the output layer.

2. Loss Calculation

The output is compared to the actual target values using a loss function (e.g., Mean Squared Error for regression, Cross-Entropy for classification).
The loss function measures how far the predicted output is from the actual target.

3. Backward Propagation

The error is propagated back through the network.
Weights and biases are adjusted to minimize the loss using optimization algorithms like Gradient Descent.

4. Training

    The processes of forward propagation, loss calculation, and backward propagation are repeated iteratively.
    The network learns to make better predictions by adjusting the weights and biases.
   
### types of neural networks

#### ANN
An Artificial Neural Network (ANN) is a computational model inspired by the way biological neural networks in the human brain process information. ANNs play a prominent role in machine learning and artificial intelligence, used for a variety of applications such as classification, regression, clustering, and more.

Application of ANN:
    
1. Image Recognition: Used in tasks like facial recognition, object detection, and image classification.
2. Natural Language Processing (NLP): Powers applications like sentiment analysis, language translation, and speech recognition.
3.  Recommendation Systems: Used by companies like Netflix and Amazon to recommend products and content.
4.  Finance: Applied in fraud detection, stock market prediction, and risk management.
5.  Healthcare: Assists in disease prediction, medical image analysis, and personalized treatment recommendations.
#### CNN
![image](https://github.com/user-attachments/assets/a010518e-faca-4ea7-b774-2bb330c07c10)

Convolutional Neural Network (CNN) is a type of deep learning model designed specifically for processing structured grid-like data, such as images. CNNs are widely used for image classification, object detection, and similar tasks due to their ability to automatically capture spatial hierarchies of features from input data through convolutions.

Application of CNN:
1. Medical Imaging: CNNs can examine thousands of pathology reports to visually detect the presence or absence of cancer cells in images
2. Audio Processing: Keyword detection can be used in any device with a microphone to detect when a certain word or phrase is spoken (“Hey Siri!”). CNNs can accurately learn and detect the keyword while ignoring all other phrases regardless of the environment.
3. Object Detection: Automated driving relies on CNNs to accurately detect the presence of a sign or other object and make decisions based on the output.
4. Synthetic Data Generation: Using Generative Adversarial Networks (GANs), new images can be produced for use in deep learning applications including face recognition and automated driving.


#### RNN
![image](https://github.com/user-attachments/assets/832841f8-24de-4ba5-8d9b-0ae01db50f86)

A Recurrent Neural Network (RNN) is a type of artificial neural network designed to recognize sequential characteristics of data and use patterns to predict the next likely scenario. RNNs are used in tasks such as time-series prediction, natural language processing, and speech recognition.
This is different from other types of neural networks as their “memory” as they take information from prior inputs to influence the current input and output.

While traditional deep learning networks assume that inputs and outputs are independent of each other, the output of recurrent neural networks depend on the prior elements within the sequence.
RNN shares parameters across each layer of the network. While feedforward networks have different weights across each node, recurrent neural networks share the same weight parameter within each layer of the network. That said, these weights are still adjusted through the processes of backpropagation and gradient descent to facilitate reinforcement learning.
# LARGE language models

#### What is it?
Large Language models are a subset of deep learning. large language model refers to large, general purpose language models that can be pretrained anf then fine tuned for specific purposes.


what does pre trained mean here? large language models are trained to solve common language problems such as text classification, question answering, document summerisation and text generation and we can add specificity by using a relatively small data regarding to a specific need.

for eg: chatgpt uses a general language model and it can be used for any field such as to summerise a text or generate code while V0 is trained on a specific data i.e code and therefore generates only code and can't summerise a text.

### Features of large language model:
1. Large - it refers to the amount of data used as well as the large number of parameters.
2. General Purpose - it should be able to solve common problems and as to train in such a way we need huge amount of data with lots of parameters, hence orgaisations such as google already have pre trained models that could be used by others. 
3. Pre- train and fine-tuned: tuning a model refers to the process of adapting a model to a new domain or set of custom use cases by trainig the model on new data.

#### LLM use cases:
1. a single model can be used for a range of activities
2. when we try to add specificity to LLM, we require less field parameters/ domain training data, such to the extent of few shot(small training data) and zero shot(no direct training).
3. Performance increases as parameters increases.

## Types of LLMs:
1. Generic (or raw) language model
![image](https://github.com/user-attachments/assets/0d457cb1-c8c3-4811-b926-e4341731347f)
It predicts the next word(specifically token) based on rhe language on the training data.
here token is the smallest unit of the words that large language model works on.
all the text we use will be broken into tokens and the large language model here essentially guesses the next token (through probablity!)
2. Instruction tuned
It gives the respose based on the instructions given in the input.
![image](https://github.com/user-attachments/assets/b590074f-1d40-407a-bebc-ca04f5c8c9d5)
3. Dialogue tuned
   Dialog- tuned models are special case od instruction tuned where requests are framed into a question. Dialogue tuning is an extenstion of instruction tuning in such a way that leads to a back and forth "conversation". There is a chain of thought reasoning that enables models to provide better outcome.


## Transformer Model


-----
# Resources used to write this article
[statquest](https://www.youtube.com/watch?v=CqOfi41LfDw&t=134s)
[Nvidia article](https://www.nvidia.com/en-in/glossary/large-language-models/)
[LLMs and Transformer](https://rpradeepmenon.medium.com/introduction-to-large-language-models-and-the-transformer-architecture-534408ed7e61)
[Neural Network Architectures & Deep Learning](https://www.youtube.com/watch?v=oJNHXPs0XDk)

completed this as a side quest!
[intro to LLM](https://www.cloudskillsboost.google/public_profiles/9ec07232-d83d-4bcb-8cac-bc290c17895d/badges/11290195)
[![Screenshot-2024-11-02-233445.png](https://i.postimg.cc/85jpY7bs/Screenshot-2024-11-02-233445.png)](https://postimg.cc/7GrrGYdy)
