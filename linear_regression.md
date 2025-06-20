# LINEAR REGRESSION FROM SCRATCH

Imagine you are watching a cricket match and you are rooting for a bats-person, you see the strike rate of the batter and based on the number of balls available you predict if he/she can score well. Here you have performed **linear regression** accidentally! if you had considered multiple criteria such as pitch and the economy of the bowler, you just performed **multiple regression** and while observing the economy of the player, if you could generally guess if they are a batter or a bowler, you just performed **classification**!

---
## What does linear regression even mean??

Linear defines the relationship between the independent and dependent variable! Woah woah hold on! what's independent and dependent variables?? 
Regression is a way to find a connection between things, so we can guess a number that defines the relationship, in a cricket match, it could be like a strike rate!
Linear regression is supervised learning i.e When we explicitly tell a program what we expect the output to be, and let it learn the rules that produce expected outputs from given inputs, we are performing supervised learning. - data is labeled and the model learns to predict the output from the input data.

    Mathematical Idea (straigt line slope):
	    Y=wX+b
	    where y -> predicted output (dependent variable)
			  X -> input feature (independent variable)
			  w -> weights a.k.a slope (learned parameter)
			  b-> bias a.k.a intercept (learned parameter)
			  
There are many places where we use linear regression - like predicting housing prices, car mileage, simple weather forecasting and more! 

### How exactly do we perform linear regression??
try to find the relationship between the dependent and independent variables - **The goal of a linear regression model is to find the slope and intercept pair that minimizes loss on average across all of the data.**
We will determine what “best” means in the following ways:

For each data point, we calculate **loss**, a number that measures how bad the model’s prediction was. It is called as error.

We can think about loss as the squared distance from the point to the line.

We do the squared distance (instead of just the distance) so that points above and below the line both contribute to total loss in the same way.loss if closer to 0 is considered to be good.

We move in the direction that decreases our loss the most. The process by which we do this is called **gradient descent.**

the process:
1.  **Start with a random formula** (hypothesis).
    
2.  Use it to **predict**.
    
3.  Check how **wrong** the predictions are.
    
4.  **Improve the formula** using gradient descent.
    
5.  Repeat until the formula gives good predictions.
    

----------

---
## Let us perform Linear regression from scratch!
> [click here for the colab](https://colab.research.google.com/drive/186G42nO5pVLUhPyzDSUpJ2rAn1tRZZsD)

>[Kaggle Dataset link](https://www.kaggle.com/datasets/snmahsa/student-score-suitable-for-linear-regression/code)
The dataset here shows the relationship between the number of hours a student studies the marks the student obtains.
#### Before we get started
Where to code?? We use Jupyter notebooks which is local to the machine or we can use Google collabs or kaggle notebook!
>Jupyter notebook tutorial
>[Google colabs](https://www.youtube.com/watch?v=TqqkZLHoY0o)
>[kaggle notebook](https://www.youtube.com/watch?v=0SiU91aBhdU)

### Libraries!
Libraries are a collection of ready-made functions that the programmer could use, python has several libraries tailored for AIML applications but here we will be using the following libraries:
-   **Pandas** – Used for **data manipulation and analysis** using tables (DataFrames).
-   **Matplotlib** – Used for **creating plots and graphs** to visualize data.
-   **NumPy** – Provides **fast mathematical operations** on arrays and matrices.
-   **Scikit-learn (sklearn)** – Offers **ready-to-use machine learning tools** like regression, classification and more.
> [watch tutorials on other libraries available for AI-ML in python!](https://www.youtube.com/watch?v=vVg7WrelMeA)
#### Step 1: Load the libraries 
and give them nick names!

    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
#### Step 2: Load the dataset
    df = pd.read_csv('/content/rounded_hours_student_scores.csv')
    df.head()
Dataset here is obtained from Kaggle, a website that  offers data science competitions, datasets, and more. 
#### Step 3: Visualize the data!

    plt.scatter(df['Hours'], df['Scores'], color='blue', alpha=0.5)
    plt.xlabel('Hours')
    plt.ylabel('Scores')
    plt.title('Hours vs Scores')
    plt.show()
Using matplotlib, we visualize the data and it should look something like this with hours on the x axis and scores on the y axis:
[![plain.png](https://i.postimg.cc/d3KVkrnH/plain.png)](https://postimg.cc/jwZbBWpP)

#### Step 4: Independent and dependent variable data

    X = df[['Hours']].values
    y = df['Scores'].values
    print(X)
    print(y)

X is a 2D NumPy array of shape (n_samples, n_features). It is the inputs or features you give to the model
y = actual outputs (what you want the model to learn to predict)
#### Step 5: Hypothesis function - aka the statistical astrologer

    def  h(X, w):
	    return w[1] * X[:,  0] + w[0]

hypothesis function is used to make predictions on the data set. In linear regression, the linear part comes into picture here as we use linear slope formula y = mc + b.
here 
-   X[:, 0] → takes the first feature column from input matrix `X` (i.e., all values of `x₁`)
    
-   w[1]→ is the **weight(slope(** for the feature `x`
    
-   w[0] → is the **bias (intercept)** term

#### Step 7: Mean Errors - less error the better!

    def  meanSquareError(X, y, w):
	    predictions = h(X, w)
		return np.mean((predictions - y)**2)

We make predictions and then we determine if our predictions are right, for this we use performance metrics to see how **off** our predictions are compared to the data.
> *Side quest: learn other types of performance metrics and try them out instead of this function*
#### Step 8: Gradient Descent - OPTIMIZE!!
[![Screenshot-2025-06-20-235619.png](https://i.postimg.cc/QxYgGzHX/Screenshot-2025-06-20-235619.png)](https://postimg.cc/tYPnF23f)
[![Screenshot-2025-06-21-000541.png](https://i.postimg.cc/DfbvrjXS/Screenshot-2025-06-21-000541.png)](https://postimg.cc/hhcRg0Mn)
Gradient descent helps you correct your aim (weights) **step by step** until you're accurate (low error).

    def  gradientDescent(X,y,w,lr):
   Inputs:
      X: input data (e.g., hours studied
      y: actual results (e.g., scores)
      w: current weights [w₀, w₁] aka slope and intercept
      lr: learning rate (how fast to update weights)
      
	    m = len(y) #no. of rows
	    y_pred = h(X,w) #preict using hypothesis function
	    error = y_pred -y #find the error
	    dw0= (1/m)* np.sum(error)
	    
This is the derivative (slope) of the loss with respect to w0 (bias).It tells you how the bias affects the overall error.
	  
	    dw1= (1/m)*np.sum(error *X[:,0])
This is the derivative (slope) of the loss with respect to w1 (slope). It tells you how the input x affects the error.

	    w[0] -=lr*dw0
	    w[1]-=lr *dw1
learning steps : here we subtract the gradient scaled by the learning rate. This means: _“step in the opposite direction of the error slope”_ to reduce the error.

	    return w


Output: updated weights
 

#### Step 9: Iterate Print Iterate Print Iterate

    def  plot_iterations(X,y,w):
	    plt.scatter(X,y,color = 'blue', alpha=0.5)
		  plt.plot(X, h(X,w), color='red')
	    plt.xlabel('Hours')
	    plt.ylabel('Scores')
	    plt.title('Hours vs Scores')
	    plt.show()
	    
    w = [0.0,  0,0]
    lr = 0.01
    epochs = 1000
    for epoch in  range(epochs):
	    w = gradientDescent(X,y,w,lr)
	    if epoch % 50 ==0:
		    plot_iterations(X,y,w)
		    print(f"epoch {epoch}: Mean Squared Error = {meanSquareError(X,y,w)}")

>*Side quest : here lr and w are hyperparameter, i.e an instruction to the model on how to learn, try to find more about hyperparameter tuning!*
>
![ezgif-com-animated-gif-maker.gif](https://i.postimg.cc/sxw2XLT6/ezgif-com-animated-gif-maker.gif)

    epoch 0: Mean Squared Error = 1109.2734526084548
    epoch 50: Mean Squared Error = 197.54302102403898
    epoch 100: Mean Squared Error = 168.159286967511
    epoch 150: Mean Squared Error = 143.4184998763084
    epoch 200: Mean Squared Error = 122.58702404295595
    epoch 250: Mean Squared Error = 105.0471461160386
    epoch 300: Mean Squared Error = 90.2787581174024
    epoch 350: Mean Squared Error = 77.84393474023688
    epoch 400: Mean Squared Error = 67.37394760043648
    epoch 450: Mean Squared Error = 58.55833137633434
    epoch 500: Mean Squared Error = 51.13567761630932
    epoch 550: Mean Squared Error = 44.88588322410555
    epoch 600: Mean Squared Error = 39.623623767093044
    epoch 650: Mean Squared Error = 35.192858072231026
    epoch 700: Mean Squared Error = 31.462201155152357
    epoch 750: Mean Squared Error = 28.32102827636842
    epoch 800: Mean Squared Error = 25.67619459862188
    epoch 850: Mean Squared Error = 23.449273173765313
    epoch 900: Mean Squared Error = 21.574229357508965
    epoch 950: Mean Squared Error = 19.99546269172615
you can observe how the mean square error keeps decreasing through the iterations as we find the best linear relation between the independent and dependent variable.
>epochs refers to one complete pass of the entire training dataset through the learning algorithm a.k.a a cycle through training data
#### Step 10: Finally- we measure and compare!
The output we obtain is:

[![iti.png](https://i.postimg.cc/4yC2xskz/iti.png)](https://postimg.cc/ftKfH4bL)

    Mean Squared Error: 18.69
    Root Mean Squared Error: 4.32
    Mean Absolute Error: 3.54
    R² Score: 0.4149

Let us see if the how right are the predictions by performance metrics and also by comparing it with the sk learn linear regression model
[![sk.png](https://i.postimg.cc/CxRXFv9H/sk.png)](https://postimg.cc/w1d02kq7)

    Mean Squared Error: 11.58
    Root Mean Squared Error: 3.40
    Mean Absolute Error: 2.50
    R² Score: 0.6374


Why is there such a difference??

1, Feature scaling is missing in our code that causes slower or uneven learning.

2. scikit-learn uses optimized solvers and not basic gradient descent leading to faster convergence.

3. Bias (intercept) handling differs which can shift predictions.

4. Learning rate and epochs in our code may not be ideal for convergence.

5. scikit-learn has higher numerical precision, making its results more stable and accurate.

> Convergence: the training process of the model  has obtained on the best values for the weights, and further updates won’t significantly reduce the error.
> 
Hope this article is helpful in understanding the basic machinery of linear regression :)

---
## Resources more to explore!!
1. [Codeacademy linear regression tutorial](https://www.codecademy.com/enrolled/courses/machine-learning-introduction-with-regression)
2. statquest youtube videos
