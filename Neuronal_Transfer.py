import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
from nst_utils import *
import numpy as np
import tensorflow as tf

"""
DEEP LEARNING AND ART -- NEURONAL STYLE TRANSFER


In this program, you will:

Implement the neural style transfer algorithm
Generate novel artistic images using your algorithm

Most of the algorithms you've studied optimize a cost function to get a set of parameter values

In Neural Style Transfer, you'll optimize a cost function to get pixel values!

"""

'''
Neural Style Transfer (NST) is one of the most fun techniques in deep learning:

It merges two images, namely, a "content" image (C) and a "style" image (S), to create a "generated" image (G)

The generated image G combines the "content" of the image C with the "style" of image S

In this example, you are going to generate:
-- an image of the Louvre museum in Paris (content image C)
-- mixed with a painting by Claude Monet, a leader of the impressionist movement (style image S)

                        ----------------------------------------------------

Neural Style Transfer (NST) uses a previously trained convolutional network, and builds on top of that
The idea of using a network trained on a different task and applying it to a new task is called {{ transfer learning }}

Following the original NST paper (https://arxiv.org/abs/1508.06576), we will use the VGG network
---> Specifically, we'll use VGG-19, a 19-layer version of the VGG network
---> This model has already been trained on the very large ImageNet database
------> It has learned to recognize a variety of low level features in the earlier layers
------> It has also learned to recognize a variety of high level features in deeper layers

Run the following code to load parameters from the VGG model. This may take a few seconds
'''

model = load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat")
print(model)

'''
The model is stored in a python dictionary
--> Each variable name is the key and the corresponding value is a tensor containing that variable's value

To run an image through this network, you just have to feed the image to the model

    In TensorFlow, you can do so using the tf.assign function
    
    In particular, you will use the assign function like this:
        --> model["input"].assign(image)
        --> This assigns the image as an input to the model
        
After, if you want to access the activations of a particular layer, say layer 4_2 when the network is run on this image
    --> you would run a TensorFlow session on the correct tensor conv4_2, as follows:
    ------> sess.run(model["conv4_2"])

                        ----------------------------------------------------

NEURONAL STYLE TRANSFER (NST)

We will build the NST algorithm in three steps:

1. Build the content cost function      --->    Jcontent(C,G) 
2. Build the style cost function        --->    Jstyle(S,G)
3. Put it together to get               --->    J(G) = [ α * Jcontent(C,G) ]  +  [ β * Jstyle(S,G) ]

In our running example, the content image C will be the picture of the Louvre Museum in Paris
Run the code below to see a picture of the Louvre.

'''

content_image = scipy.misc.imread("images/content_oliver.png")
imshow(content_image)
plt.show()

'''

The content image (C) shows the Louvre museum surrounded by old Paris buildings, against a sunny sky with a few clouds

                        **********************************************************************
                      How do you ensure the generated image G matches the content of the image C?
                        **********************************************************************
-- The earlier (shallower) layers of a ConvNet tend to detect lower-level features such as edges and simple textures
-- The later (deeper) layers tend to detect higher-level features such as complex textures as well as object classes
                        **********************************************************************
                        
We would like the "generated" image G to have similar content as the input image C

Suppose you have chosen some layer's activations to represent the content of an image
In practice, you'll get the most visually pleasing results if you choose a layer in the middle of the network
    --> Neither too shallow nor too deep
    -----> After you have finished this exercise, feel free to come back and experiment with using different layers

So, suppose you have picked one particular hidden layer to use
Now, set the image C as the input to the pretrained VGG network, and run forward propagation

Let  a(C)  be the hidden layer activations in the layer you had chosen
    --> In lecture, we had written this as  a[l](C),  but here we'll drop the superscript  [l]  to simplify the notation
    --> This will be a  nH×nW×nC  tensor

Repeat this process with the image G: Set G as the input, and run forward progation
Let a(G) be the corresponding hidden layer activation

                        **********************************************************************
                                     We will define the content cost function as:
                        **********************************************************************

                Jcontent(C,G) = { 1 / [ 4 × nH × nW× nC ] } * ∑ <<over all entries of (a(C) − a(G))^2 >>
                
                        Here:   nH, nW and nC  are the height, width and number of channels of the hidden layer ...
                                ... they appear in a normalization term in the cost
                               
                --> Note:  a(C)  &  a(G)  are the volumes corresponding to a hidden layer's activations
    
In order to compute the cost  Jcontent(C,G), it might also be convenient to unroll these 3D volumes into a 2D matrix
--> Technically this unrolling step isn't needed to compute  Jcontent
--> However, it will be practice for when we carry out a similar operation later in computing the style constant  Jstyle


                        **********************************************************************

                                Exercise: Compute the "content cost" using TensorFlow

                        **********************************************************************


Instructions:
--------------

1. Retrieve dimensions from a_G:
    --> To retrieve dimensions from a tensor X, use: X.get_shape().as_list()

2. Unroll a_C and a_G as explained in the picture above

3. Compute the content cost:

'''


# GRADED FUNCTION: compute_content_cost

def compute_content_cost(a_C, a_G):
    """
    Computes the content cost

    Arguments:
    a_C -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image C
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image G

    Returns:
    J_content -- scalar that you compute using equation 1 above.
    """

    # Retrieve dimensions from a_G (≈1 line)
    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    # Reshape a_C and a_G (≈2 lines)
    a_C_unrolled = tf.reshape(a_C, [m, n_H * n_W, n_C])
    a_G_unrolled = tf.reshape(a_G, [m, n_H * n_W, n_C])

    # compute the cost with tensorflow (≈1 line)
    J_content = (1 / (4 * n_H * n_W * n_C)) * tf.reduce_sum(tf.square(tf.subtract(a_C, a_G)))

    return J_content


tf.reset_default_graph()

with tf.Session() as test:
    tf.set_random_seed(1)
    a_C = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
    a_G = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
    J_content = compute_content_cost(a_C, a_G)
    print("\nJ_content = " + str(J_content.eval()))

'''
What you should remember:

The content cost takes a hidden layer activation of the neural net, and measures the difference between  a(C)  and  a(G)
q
When we minimize the content cost later, this will help make sure  G  has similar content as  C 

'''

'''
For our running example, we will use the following style image (shown next)
'''

style_image = scipy.misc.imread("images/style_colville.png")
imshow(style_image)
plt.show()

'''
This painting was painted in the style of impressionism

Lets see how you can now define a "style" const function  Jstyle(S,G)

                               *******************************************************
                                                    STYLE MATRIX
                               *******************************************************

The style matrix is also called a "Gram Matrix" In linear algebra
The Gram matrix G, of a set of vectors  (v1,…,vn),  is the matrix of dot products
---> The dot product entries are  G[i][j] = v[i][T] * v[j] = np.dot(vi,vj)
---> In other words,  jGij  compares how similar  vi  is to  vj
    --- If they are highly similar, you would expect them to have a large dot product, and thus  Gij  to be large

NOTE:   ... There is an unfortunate collision in the variable names used here ...
NOTE:   ... We are following common terminology used in the literature ...
NOTE:   ... but  G  is used to denote the Style matrix (or Gram matrix) as well as to denote the generated image  G ...
NOTE:   ... We will try to make sure which  G  we are referring to is always clear from the context ...


In NST, you can compute the Style matrix by multiplying the "unrolled" filter matrix with their transpose:

The result is a matrix of dimension  (nC,nC)  where  nC  is the number of filters
The value  G[i][j]  measures how similar the activations of filter  [i]  are to the activations of filter  [j]

One important part of the gram matrix is the diagonal elements such as  G[i][i]  also measures how active filter [i] is
    
    i.e. Suppose filter  [i]  is detecting vertical textures in the image ...
         ... then  G[i][i]  measures how common vertical textures are in the image as a whole:
            --> If  G[i][i]  is large, this means that the image has a lot of vertical texture


By capturing the prevalence of different types of features ( G[i][i] ) ...
... as well as how much different features occur together ( G[i][j] ) ...
... the Style matrix  G  measures the style of an image

                
                
                        **********************************************************************

                Exercise: Using TensorFlow, implement a function that computes the Gram matrix of a matrix A

                        **********************************************************************


Instructions:
--------------

The formula is: 
        
        The gram matrix of A is  G[A] = A*Atranspose
        
'''


# Gram_Matrix G - STYLE MATRIX
def gram_matrix(A):
    """
    Argument:
    A -- matrix of shape (n_C, n_H*n_W)

    Returns:
    GA -- Gram matrix of A, of shape (n_C, n_C)
    """

    GA = tf.matmul(A, A, transpose_a=False, transpose_b=True)

    return GA


tf.reset_default_graph()

with tf.Session() as test:
    tf.set_random_seed(1)
    A = tf.random_normal([3, 2 * 1], mean=1, stddev=4)
    GA = gram_matrix(A)

    print("\nGA = \n" + str(GA.eval()))

'''
STYLE COST

After generating the Style matrix (Gram matrix) your goal will be to:
-- Minimize the distance between the Gram matrix of the "style" image S and that of the "generated" image G

    For now, we are using only a single hidden layer  a[l], & the corresponding style cost for this layer is defined as:
        ---SUMS GO FROM i/j TO n[c]---
                  J[l]style(S,G)=[ 1 / {4 × n[C]^2 ×(n[H] × n[W])^2 }   *  ∑∑ ( G(S)[i][j] − G(G)[i][j] )^2
 
where  G(S)  and  G(G)  are respectively the Gram matrices of the "style" image and the "generated" image
---> They are computed using the hidden layer activations for a particular hidden layer in the network

Exercise: Compute the style cost for a single layer

Instructions: The 4 steps to implement this function are:

1. Retrieve dimensions from the hidden layer activations a_G:
   ---> To retrieve dimensions from a tensor X, use: X.get_shape().as_list()

2. Unroll the hidden layer activations a_S and a_G into 2D matrices, as explained in the picture above.

3. Compute the Style matrix of the images S and G. (Use the function you had previously written.)

4. Compute the Style cost:

'''


def compute_layer_style_cost(a_S, a_G):
    """
    Arguments:
    a_S -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image S
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image G

    Returns:
    J_style_layer -- tensor representing a scalar value, style cost defined above by equation (2)
    """

    # Retrieve dimensions from a_G (≈1 line)
    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    # Reshape the images to have them of shape (n_C, n_H*n_W) (≈2 lines)
    a_S = tf.transpose(tf.reshape(a_S, [(n_H * n_W), n_C]))
    a_G = tf.transpose(tf.reshape(a_G, [(n_H * n_W), n_C]))

    # Computing gram_matrices for both images S and G (≈2 lines)
    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)

    # Computing the loss (≈1 line)
    J_style_layer = (1 / (4 * tf.square(tf.to_float(n_C)) * tf.square(tf.to_float(n_W * n_H)))) * \
                    (tf.reduce_sum(tf.square(tf.subtract(GS, GG))))

    return J_style_layer


tf.reset_default_graph()

with tf.Session() as test:
    tf.set_random_seed(1)
    a_S = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
    a_G = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
    J_style_layer = compute_layer_style_cost(a_S, a_G)

    print("\nJ_style_layer = " + str(J_style_layer.eval()))

'''
So far you have captured the style from only one layer
We'll get better results if we "merge" style costs from several different layers

After completing this exercise, feel free to come back and experiment with different weights... 
-- This will change the generated image  G
-- But for now, this is a pretty reasonable default: Shown below...

'''

STYLE_LAYERS = [
    ('conv1_1', 0.2),
    ('conv2_1', 0.2),
    ('conv3_1', 0.2),
    ('conv4_1', 0.2),
    ('conv5_1', 0.2)]

'''
You can combine the style costs for different layers as follows:

            Jstyle(S,G)=∑(over-layers of -->) λ[l] * J[l][style](S,G)
            
    Where the values for  λ[l]λ[l]  are given in STYLE_LAYERS.


We've implemented a compute_style_cost(...) function...

It calls the compute_layer_style_cost(...) several times, and weights their results using the values in STYLE_LAYER
Read over it to make sure you understand what it's doing

'''


def compute_style_cost(model, STYLE_LAYERS):
    """
    Computes the overall style cost from several chosen layers

    Arguments:
    model -- our tensorflow model
    STYLE_LAYERS -- A python list containing:
                        - the names of the layers we would like to extract style from
                        - a coefficient for each of them

    Returns:
    J_style -- tensor representing a scalar value, style cost defined above by equation (2)
    """

    # initialize the overall style cost
    J_style = 0

    for layer_name, coeff in STYLE_LAYERS:
        # Select the output tensor of the currently selected layer
        out = model[layer_name]

        # Set a_S to be the hidden layer activation from the layer we have selected, by running the session on out
        a_S = sess.run(out)

        # Set a_G to be the hidden layer activation from same layer. Here, a_G references model[layer_name]
        # and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
        # when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
        a_G = out

        # Compute style_cost for the current layer
        J_style_layer = compute_layer_style_cost(a_S, a_G)

        # Add coeff * J_style_layer of this layer to overall style cost
        J_style += coeff * J_style_layer

    return J_style

'''

                                                --- Note --- 

In the inner-loop of the for-loop above, a_G is a tensor and hasn't been evaluated yet
It will be evaluated and updated at each iteration when we run the TensorFlow graph in model_nn() below


                                        --- What you should remember ---

The style of an image can be represented using the Gram matrix of a hidden layer's activations

However, we get even better results combining this representation from multiple different layers

This is in contrast to the content representation, where usually using just a single hidden layer is sufficient

Minimizing the style cost will cause the image  Gto follow the style of the image  S



Finally, let's create a cost function that minimizes both the style and the content cost

    The formula is:             J(G)= α * J[content](C,G) + β * J[style](S,G)
 
Exercise: Implement the total cost function which includes both the content cost and the style cost.

'''


def total_cost(J_content, J_style, alpha=10, beta=40):
    """
    Computes the total cost function

    Arguments:
    J_content -- content cost coded above
    J_style -- style cost coded above
    alpha -- hyperparameter weighting the importance of the content cost
    beta -- hyperparameter weighting the importance of the style cost

    Returns:
    J -- total cost as defined by the formula above.
    """

    J = alpha * J_content + beta * J_style

    return J


tf.reset_default_graph()

with tf.Session() as test:
    np.random.seed(3)
    J_content = np.random.randn()
    J_style = np.random.randn()
    J = total_cost(J_content, J_style)
    print("\nJ = " + str(J))

'''
What you should remember:
-- The total cost is a linear combination of the content cost  Jcontent(C,G)  and the style cost  Jstyle(S,G) 
-- α  and  β  are hyperparameters that control the relative weighting between content and style

Finally, let's put everything together to implement Neural Style Transfer!

Here's what the program will have to do:

1. Create an Interactive Session
2. Load the content image
3. Load the style image
4. Randomly initialize the image to be generated
5. Load the VGG16 model
6. Build the TensorFlow graph:
    -- Run the content image through the VGG16 model and compute the content cost
    -- Run the style image through the VGG16 model and compute the style cost
    -- Compute the total cost
    -- Define the optimizer and the learning rate
7. Initialize the TensorFlow graph and run it for a large number of iterations, updating the generated image at every step.

Lets go through the individual steps in detail...

You've previously implemented the overall cost J(G)...

We'll now set up TensorFlow to optimize this with respect to G

To do so, your program has to reset the graph and use an "Interactive Session"

Unlike a regular session, the "Interactive Session" installs itself as the default session to build a graph
This allows you to run variables without constantly needing to refer to the session object, which simplifies the code

Lets start the interactive session:

'''

# Reset the graph
tf.reset_default_graph()

# Start interactive session
sess = tf.InteractiveSession()

'''
Let's load, reshape, and normalize our "content" image (the Louvre museum picture):
'''

content_image = scipy.misc.imread("images/content_oliver.png")
content_image = reshape_and_normalize_image(content_image)

'''
Let's load, reshape and normalize our "style" image (Claude Monet's painting):
'''

style_image = scipy.misc.imread("images/style_colville.png")
style_image = reshape_and_normalize_image(style_image)

'''
Now, we initialize the "generated" image as a noisy image created from the content_image
i.e. The pixels of the generated image are mostly noise but still slightly correlated with the content image
    -- This will help the content of the "generated" image more rapidly match the content of the "content" image
'''

generated_image = generate_noise_image(content_image)
imshow(generated_image[0])

'''
Next, as explained in part (2), let's load the VGG16 model
'''

model = load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat")

'''
To compute the content cost, we will now assign a_C and a_G to be the appropriate hidden layer activations
We will use layer conv4_2 to compute the content cost

The code below does the following:

    1. Assign the content image to be the input to the VGG model.
    2. Set a_C to be the tensor giving the hidden layer activation for layer "conv4_2".
    3. Set a_G to be the tensor giving the hidden layer activation for the same layer.
    4.Compute the content cost using a_C and a_G.
'''

# Assign the content image to be the input of the VGG model.
sess.run(model['input'].assign(content_image))

# Select the output tensor of layer conv4_2
out = model['conv4_2']

# Set a_C to be the hidden layer activation from the layer we have selected
a_C = sess.run(out)

# Set a_G to be the hidden layer activation from same layer. Here, a_G references model['conv4_2']
# and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
# when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
a_G = out

# Compute the content cost
J_content = compute_content_cost(a_C, a_G)

'''
                                                    ---Note---
                                                    
At this point, a_G is a tensor and hasn't been evaluated

It will be evaluated and updated at each iteration when we run the Tensorflow graph in model_nn() below
'''

# Assign the input of the model to be the "style" image
sess.run(model['input'].assign(style_image))

# Compute the style cost
J_style = compute_style_cost(model, STYLE_LAYERS)

'''
Exercise: 

Now that you have J_content and J_style, compute the total cost J by calling total_cost(). Use alpha = 10 and beta = 40

'''

J = total_cost(J_content, J_style, alpha = 10, beta = 40)

'''
You'd previously learned how to set up the Adam optimizer in TensorFlow
Lets do that here, using a learning rate of 2.0
'''

# define optimizer (1 line)
optimizer = tf.train.AdamOptimizer(2.0)

# define train_step (1 line)
train_step = optimizer.minimize(J)

'''
Exercise: 

Implement the model_nn() function
-- initializes the variables of the tensorflow graph
-- assigns the input image (initial generated image) as the input of the VGG16 model
-- runs the train_step for a large number of steps

'''


def model_nn(sess, input_image, num_iterations=600):
    # Initialize global variables (you need to run the session on the initializer)
    sess.run(tf.global_variables_initializer())

    # Run the noisy input image (initial generated image) through the model. Use assign().
    sess.run(model["input"].assign(input_image))

    for i in range(num_iterations):

        # Run the session on the train_step to minimize the total cost
        sess.run(train_step)

        # Compute the generated image by running the session on the current model['input']
        generated_image = sess.run(model["input"])

        # Print every 20 iteration.
        if i % 20 == 0:
            Jt, Jc, Js = sess.run([J, J_content, J_style])
            print("Iteration " + str(i) + " :")
            print("total cost = " + str(Jt))
            print("content cost = " + str(Jc))
            print("style cost = " + str(Js))

            # save current generated image in the "/output" directory
            save_image("output/" + str(i) + ".png", generated_image)

    # save last generated image
    save_image('output/generated_image.jpg', generated_image)

    return generated_image


model_nn(sess, generated_image)

'''
Great job on completing this assignment! You are now able to use Neural Style Transfer to generate artistic images
This is also your first time building a model in which the optimization algorithm updates the pixel values
-- Rather than the neural network's parameter
 
Deep learning has many different types of models and this is only one of them!

                                            What you should remember:


-- Neural Style Transfer is an algorithm that given a content image C and a style image S can generate an artistic image

-- It uses representations (hidden layer activations) based on a pretrained ConvNet

-- The content cost function is computed using one hidden layer's activations

-- The style cost function for one layer is computed using the Gram matrix of that layer's activations
    ---> The overall style cost function is obtained using several hidden layers.

-- Optimizing the total cost function results in synthesizing new images

   
                                                    NOTE:
                                   -----------------------------------------
                                     Feel free to try with your own images
                                   ----------------------------------------- 
   
        To do so, go back to part 4 and change the content image and style image with your own pictures
        In detail, here's what you should do:
                
            Click on "File -> Open" in the upper tab of the notebook
            Go to "/images" and upload your images (requirement: (WIDTH = 400, HEIGHT = 300)) -- this was 300x255
            Rename them "my_content.png" and "my_style.png" for example
        
            Change the code in part (3.4) from :
                    content_image = scipy.misc.imread("images/louvre.jpg")
                    style_image = scipy.misc.imread("images/claude-monet.jpg")
        
            To the code:
                    content_image = scipy.misc.imread("images/my_content.jpg")
                    style_image = scipy.misc.imread("images/my_style.jpg")

            Rerun the cells (you may need to restart the Kernel in the upper tab of the notebook)


            You can also tune your hyperparameters:            
                    Which layers are responsible for representing the style? STYLE_LAYERS
                    How many iterations do you want to run the algorithm? num_iterations
                    What is the relative weighting between content and style? alpha/beta
                                            
'''