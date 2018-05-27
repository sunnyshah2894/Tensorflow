import tensorflow as tf
import numpy as np


# training set. Contains a row of size 5 per train example. The row is same as a sentence, with words replaced
# by its equivalent unique index. The below dataset contains 6 unique words numbered 0-5. Ideally the word vector for
# 4 and 5 indexed words should be same.
X_train = np.array([[0,1,4,2,3],[0,1,5,2,3]])

# output dummy for testing purpose
y_train = np.array([0,1])


# Create the embeddings
with tf.name_scope("embeddings"):
    # Initiliaze the embedding vector by randomly distributing the weights.
    embedding = tf.Variable(tf.random_uniform((6,
                              3), -1, 1))
    # create the embedding layer
    embed = tf.nn.embedding_lookup(embedding, X_train)

    # So that we can apply a convolution 2d operations on top the expanded single channel embedded vectors
    embedded_chars_expanded = tf.expand_dims(embed, -1)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer());
    result,result_expanded = sess.run([embed,embedded_chars_expanded]);
    print(result_expanded.shape)
    print(result)
    print(result_expanded)


# OUTPUT
# result
# [[[ 0.89598155  0.4275496   0.00858593]
#   [ 0.21602225 -0.44228792 -0.20533657]
#   [ 0.9624436  -0.99176955  0.15964746]
#   [-0.29004955  0.470721    0.00804782]
#   [ 0.7497003   0.6044979  -0.5612638 ]]
#
#  [[ 0.89598155  0.4275496   0.00858593]
#   [ 0.21602225 -0.44228792 -0.20533657]
#   [-0.48809385 -0.55618596 -0.73995876]
#   [-0.29004955  0.470721    0.00804782]
#   [ 0.7497003   0.6044979  -0.5612638 ]]]

# result_expanded - has a dimension of (2,5,3,1)
# [[[[-0.45975637]
#    [-0.5756638 ]
#    [ 0.7002065 ]]
#
#   [[ 0.2708087 ]
#    [ 0.7985747 ]
#    [ 0.57897186]]
#
#   [[ 0.6642673 ]
#    [ 0.6548476 ]
#    [ 0.00760126]]
#
#   [[-0.7074845 ]
#    [ 0.5100081 ]
#    [ 0.7232883 ]]
#
#   [[ 0.19342017]
#    [-0.46509933]
#    [ 0.8361807 ]]]
#
#
#  [[[-0.45975637]
#    [-0.5756638 ]
#    [ 0.7002065 ]]
#
#   [[ 0.2708087 ]
#    [ 0.7985747 ]
#    [ 0.57897186]]
#
#   [[-0.90803576]
#    [ 0.75451994]
#    [ 0.8864901 ]]
#
#   [[-0.7074845 ]
#    [ 0.5100081 ]
#    [ 0.7232883 ]]
#
#   [[ 0.19342017]
#    [-0.46509933]
#    [ 0.8361807 ]]]]