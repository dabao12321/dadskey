## The laziest man's guide to Machine Learning

Here's what I used to go from a limited knowledge of Java to a limited knowledge of machine learning.

### Python
There's a lot of useful stuff for ML with Python, and it's not hard to pick up. If you don't want to go through an actual tutorial, just know these:
- whitespace matters, esp. tabs
- instead of {} for nested whatevers, use :
- dynamically typed
- instead of ```for(var i = 0; i < a[].length; i++)```, use ```for item, index in enumerate(a[])```. So eZ. So nice.

For an actual tutorial, check the repo for the Python crash course docs. 


### Machine Learning concepts

Explaining all these will take too long here. Summaries below, in the order I learned them. 

#### Partial Derivatives and Gradients
Very important to understand, less important to know how to actually calculate. A partial derivative tell you how much the function is changing in only one direction (i.e. "slope"). A gradient is the vector pointing in the direction of the function's greatest change. 

Ex. if your function is 3D, looks like a hill, then the gradient points you to the top of the hill. 

#### Optimization and Cost Functions
Your model will always output a __cost function__, indicating the degree of error. Cost functions can vary, but they essentially calculate the `correct value - machine produced value`. 

You can imagine it plotting a set of weights `w` on the x-axis against a set of biases `b` on the y-axis, so that `f(w, b) = cost`. If you modify a bunch of weights and biases, you'll generate the cost function. You obviously want to minimize cost, so that's where gradients come in. 

Gradients point you in the direction of greatest positive change. You want the smallest amount of error. So, go in the opposite direction of the gradient. This is called __Gradient Descent Optimization__. 

There's many other types of optimizers, that are more complicated and fun. This one works well and is easy to understand.

The next step in optimizing is __Stocastic Gradient Descent__. "Stocastic" can be thought of as "in steps". Instead of, say, following your gradients in whichever direction they point in, SGD can only move along the x and y axes. SGD takes steps. 

#### Hyperparameters and Cross Validation
The only thing you really have control over in your network is the __hyperparameters__: variables you define before the network even starts to train. Ex. weights and biases are not hyperparameters. The __batch size__ of your training data (how many datums you train on before updating your network's parameters) is a hyperparameter. 

Without knowing much math, a lot of picking hyperparameters is educated guesswork. Ways to efficiently do so is reading about networks made by people smarter than you. That gives you context, a sort of range to play around in. Once you've set your hyperparams, you have to test it to see if they are useful. 

Testing is done with Cross Validation.
1. Split your data in a number of groups, say 5. 
2. 3 groups are training sets. 
3. 1 group is a validation set-- it's not training on this data. This is to make sure that the network can adapt to new data, and is not just showing improvement on the training set. If there is no improvement on the validation set, stop training.
4. The last group is the test set-- use completely new data. The result is the "official" accuracy and loss scoring of the network. 

#### Basic runthrough of a neural net

Let's train.

1. Input each  value into the first layer of the net, the "input layer".
2. Each node in the first layer connects to each node in the second layer. Each edge can be though of as `value * edge_weight`, where the `edge_weight` is initial randomized. 
3. Use an __activation function__, which makes sure that your data is not linear. The most popular one, ReLU, simply sets all negative values to be zero. 
3. Continue to do the same til the last layer. This last layer outputs probabilities for each potential classification.
4. Compare the machine's prediction to the actual answer.
5. Calculate lost, updating what the cost function looks like.
6. Continue to train on all data in the batch, building more of the cost function.
7. Updates the weights and biases based on the cost function.
8. Repeat with the next batch.

#### Backpropogation

Probably, the question is on step 7: how the network updates the parameters. This is called __backpropogation__: from the loss score, it traces from the output layer back to the input to modify the weights. 

You need calculus to understand this. Luckily, you won't need to implement this yourself.

### Python's ML Libraries

These libraries make lives much easier. 

Numpy: Low-level. For matrix manipulation and general mathematics. Probably won't need unless you're implementing the math-heavy functions.

Matplotlib: Graphing matrices. Not necessary, but it can help you see what's going on in the network. 

__Tensorflow__: The holy grail. High-level. Has enough built-in functions for you to build a network without dealing with excessive math. It will be more like the 8 steps I described above.

Link: https://www.tensorflow.org/

Ex. This is linear regression with TF.

```
# Basic Linear Regression Training
# Amanda Li

import tensorflow as tf
LOGDIR = "./tmp/linear_regression"

# Model parameters
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
# Model input and output
x = tf.placeholder(tf.float32)
linear_model = W * x + b
y = tf.placeholder(tf.float32)

# loss
loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares
# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# training data
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]
# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to wrong
for i in range(1000):
  sess.run(train, {x: x_train, y: y_train})

# evaluate training accuracy
curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))

writer = tf.summary.FileWriter(LOGDIR, sess.graph)
writer.add_graph(sess.graph)
```

### Onwards

This was about a week and half into my independent project syllabus. There are a variety of more topics that I find interesting, but for now, this is probably enough to be useful. 
