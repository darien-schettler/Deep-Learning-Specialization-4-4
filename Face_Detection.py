from inception_blocks_v2 import *
from keras import backend as K
from numpy import genfromtxt
from fr_utils import *
from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
import cv2
import os
import numpy as np
import pandas as pd
import tensorflow as tf

# Make Keras take channels first when looking at the structure of images
K.set_image_data_format('channels_first')

# Make it so that their is no summarization when printing numpy arrays
np.set_printoptions(threshold=np.nan)

'''
The key things you need to know are:

This network uses 1920x1080 dimensional RGB images as its input

Specifically:
    -- Inputs a face image (or batch of  m  face images) as a tensor of shape  (m,nC,nH,nW)  =  (m,3,1920,1080)
    -- Outputs a matrix of shape  (m,128)  that encodes each input face image into a 128-dimensional vector

'''

# Create the model for face images
FRmodel = faceRecoModel(input_shape=(3, 96, 96))

print("\nTotal Params:", FRmodel.count_params())

'''

So, an encoding is a good one if:

The encodings of two images of the same person are quite similar to each other
The encodings of two images of different persons are very different

The triplet loss function formalizes this
    -- It tries to "push" the encodings of two images of the same person (Anchor and Positive) closer together
    -- It does this while "pulling" the encodings of two images of different persons (Anchor, Negative) further apart
    -- Training will use triplets of images (A,P,N):
            << A >> is an "Anchor" image    -- a picture of a person
            << P >> is a "Positive" image   -- a picture of the same person as the Anchor image         (positive)
            << N >> is a "Negative" image   -- a picture of a different person than the Anchor image    (negative)
    -- To compute triplet loss we follow these 4 steps:
            1. Compute the distance between the encodings of "anchor" and "positive"
            2. Compute the distance between the encodings of "anchor" and "negative"
            3. Compute the formula per training example
            4. Compute the full formula by taking the max with zero and summing over the training examples
            
            NOTE: USEFUL FUNCTIONS: {{{ tf.reduce_sum(), tf.square(), tf.subtract(), tf.add(), tf.maximum() }}}

'''


# GRADED FUNCTION: triplet_loss

def triplet_loss(y_true, y_pred, alpha=0.2):
    """
    Implementation of the triplet loss as defined by formula (3)

    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor images, of shape (None, 128)
            positive -- the encodings for the positive images, of shape (None, 128)
            negative -- the encodings for the negative images, of shape (None, 128)

    Returns:
    loss -- real number, value of the loss
    """

    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

    # Step 1: Compute the (encoding) distance between the anchor and the positive, you will need to sum over axis=-1
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), -1)

    # Step 2: Compute the (encoding) distance between the anchor and the negative, you will need to sum over axis=-1
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), -1)

    # Step 3: subtract the two previous distances and add alpha.
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)

    # Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples.
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))

    return loss


with tf.Session() as test:
    tf.set_random_seed(1)
    y_true = (None, None, None)
    y_pred = (tf.random_normal([3, 128], mean=6, stddev=0.1, seed=1),
              tf.random_normal([3, 128], mean=1, stddev=1, seed=1),
              tf.random_normal([3, 128], mean=3, stddev=4, seed=1))
    loss = triplet_loss(y_true, y_pred)

    print("loss = " + str(loss.eval()))

# Load the model
FRmodel.compile(optimizer='adam', loss=triplet_loss, metrics=['accuracy'])
load_weights_from_FaceNet(FRmodel)

'''

Let's build a database containing one encoding vector for each person allowed to enter the happy house

To generate the encoding we use img_to_encoding(image_path, model)
    -- This basically runs the forward propagation of the model on the specified image
'''

database = {}
database["danielle"] = img_to_encoding("images/danielle.png", FRmodel)
database["younes"] = img_to_encoding("images/younes.jpg", FRmodel)
database["tian"] = img_to_encoding("images/tian.jpg", FRmodel)
database["andrew"] = img_to_encoding("images/andrew.jpg", FRmodel)
database["kian"] = img_to_encoding("images/kian.jpg", FRmodel)
database["dan"] = img_to_encoding("images/dan.jpg", FRmodel)
database["sebastiano"] = img_to_encoding("images/sebastiano.jpg", FRmodel)
database["bertrand"] = img_to_encoding("images/bertrand.jpg", FRmodel)
database["kevin"] = img_to_encoding("images/kevin.jpg", FRmodel)
database["felix"] = img_to_encoding("images/felix.jpg", FRmodel)
database["benoit"] = img_to_encoding("images/benoit.jpg", FRmodel)
database["arnaud"] = img_to_encoding("images/arnaud.jpg", FRmodel)

'''
Now, when someone shows up at your front door and swipes their ID card (thus giving you their name):
    --- You can look up their encoding in the database,
    --- And use it to check if the person standing at the front door matches the name on the ID


Exercise: 
Implement the verify() function which checks if the front-door camera picture is actually the person called "identity"

You will have to go through the following steps:

1. Compute the encoding of the image from image_path
2. Compute the distance about this encoding and the encoding of the identity image stored in the database
3. Open the door if the distance is less than 0.7, else do not open.

As presented above, you should use the L2 distance (np.linalg.norm)
    --- Note: In this implementation, compare the L2 distance, not the square of the L2 distance, to the threshold 0.7

'''


# GRADED FUNCTION: verify

def verify(image_path, identity, database, model):
    """
    Function that verifies if the person on the "image_path" image is "identity".

    Arguments:
    image_path -- path to an image
    identity -- string, name of the person you'd like to verify the identity. Has to be a resident of the Happy house.
    database -- python dictionary mapping names of allowed people's names (strings) to their encodings (vectors).
    model -- your Inception model instance in Keras

    Returns:
    dist -- distance between the image_path and the image of "identity" in the database.
    door_open -- True, if the door should open. False otherwise.
    """

    # Step 1: Compute the encoding for the image. Use img_to_encoding() see example above. (≈ 1 line)
    encoding = img_to_encoding(image_path, model)

    # Step 2: Compute distance with identity's image (≈ 1 line)
    dist = np.linalg.norm(encoding - database[identity])

    # Step 3: Open the door if dist < 0.7, else don't open (≈ 3 lines)
    if dist < 0.7:
        print("It's " + str(identity) + ", welcome home!")
        door_open = True
    else:
        print("It's not " + str(identity) + ", please go away")
        door_open = False

    return dist, door_open


verify("images/camera_0.jpg", "younes", database, FRmodel)

verify("images/camera_2.jpg", "kian", database, FRmodel)

'''

Implement a face recognition system:
    -- It takes as input an image
    -- It figures out if it is an authorized person (and if so, who) and outputs that
    
Unlike the previous face verification system, we will no longer get a person's name as another input

Exercise: Implement who_is_it()

You will have to go through the following steps:

1.Compute the target encoding of the image from image_path
2.Find the encoding from the database that has smallest distance with the target encoding.
    -- Initialize the min_dist variable to a large enough number (100)
        -> It will help you keep track of what is the closest encoding to the input's encoding
    -- Loop over the database dictionary's names and encodings. To loop use for (name, db_enc) in database.items()

        --- Compute L2 distance between the target "encoding" and the current "encoding" from the database ---
        --- If this distance is less than the min_dist, then set min_dist to dist, and identity to name ---

'''


def who_is_it(image_path, database, model):
    """
    Implements face recognition for the happy house by finding who is the person on the image_path image.

    Arguments:
    image_path -- path to an image
    database -- database containing image encodings along with the name of the person on the image
    model -- your Inception model instance in Keras

    Returns:
    min_dist -- the minimum distance between image_path encoding and the encodings from the database
    identity -- string, the name prediction for the person on image_path
    """

    encoding = img_to_encoding(image_path, model)

    # Initialize "min_dist" to a large value, say 100 (≈1 line)
    min_dist = 100

    # Loop over the database dictionary's names and encodings.
    for (name, db_enc) in database.items():

        # Compute L2 distance between the target "encoding" and the current "emb" from the database. (≈ 1 line)
        dist = np.linalg.norm(encoding - db_enc)

        # If this distance is less than the min_dist, then set min_dist to dist, and identity to name. (≈ 3 lines)
        if dist < min_dist:
            min_dist = dist
            identity = name

    if min_dist > 0.7:
        print("Not in the database.")
    else:
        print("it's " + str(identity) + ", the distance is " + str(min_dist))

    return min_dist, identity


'''

Your Happy House is running well
    --- It only lets in authorized persons, and people don't need to carry an ID card around anymore!

You've now seen how a state-of-the-art face recognition system works


Although we won't implement it here, here're some ways to further improve the algorithm:

1. Put more images of each person (under different lighting conditions, taken on different days, etc.) into the database
    --- Then given a new image, compare the new face to multiple pictures of the person (This would increase accuracy)

2.Crop the images to just contain the face, and less of the "border" region around the face
    ---- This preprocess removes some of the irrelevant pixels around the face, and also makes the algorithm more robust


What you should remember:

1. Face verification solves an easier 1:1 matching problem; face recognition addresses a harder 1:K matching problem
2. The triplet loss is an effective loss function for training a neural network to learn an encoding of a face image
3. The same encoding can be used for verification and recognition
    --- Measuring distances between two images' encodings lets you to determine if they are pictures of the same person

'''
