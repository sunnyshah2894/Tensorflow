import pandas as pd
import numpy as np
import tensorflow as tf


def cnn_model(data, labels, VOCAB_SIZE , EMBEDDING_DIMENSION=128 , MAX_SEQUENCE_LENGTH = 200 ):
    """2 layer ConvNet to predict from sequence of words to a class."""
    # Convert indexes of words into embeddings.
    # This creates embeddings matrix of [n_words, EMBEDDING_SIZE] and then
    # maps word indexes of the sequence into [batch_size, sequence_length,
    # EMBEDDING_SIZE].

    # Create the embeddings
    # Declare placeholders we'll feed into the graph


    with tf.name_scope("embeddings"):

        # Initiliaze the embedding vector by randomly distributing the weights.
        embedding = tf.Variable(tf.random_uniform((VOCAB_SIZE,EMBEDDING_DIMENSION), -1, 1))

        # create the embedding layer
        embed = tf.nn.embedding_lookup(embedding, data)

        # So that we can apply a convolution 2d operations on top the expanded single channel embedded vectors
        # Shape will be NONE , MAX_SEQUENCE_LENGTH , EMBEDDING_DIMENSION , 1
        embedded_chars_expanded = tf.expand_dims(embed, -1)

    print( embedded_chars_expanded )
    with tf.variable_scope('CNN_Layer1_2'):
        # Apply Convolution filtering on input sequence.
        conv1 = tf.layers.conv2d(
            inputs=tf.reshape(embedded_chars_expanded,[-1,MAX_SEQUENCE_LENGTH,EMBEDDING_DIMENSION,1]),
            filters=10,
            kernel_size=[10,EMBEDDING_DIMENSION], # so that the entire words are convoluted so that network learns some neighberhood relationship
            padding='SAME',
            # Add a ReLU for non linearity.
            activation=tf.nn.relu)
        # Max pooling across output of Convolution+Relu.
        pool1 = tf.layers.max_pooling2d(
            conv1,
            pool_size=[1,2],
            strides=[1,2],
            padding='VALID')
        # Transpose matrix so that n_filters from convolution becomes width.
        #pool1 = tf.transpose(pool1, [0, 1, 3, 2])
        drop = tf.layers.dropout(pool1, rate=0.25)
    with tf.variable_scope('CNN_Layer2_2'):
        # Second level of convolution filtering.
        conv2 = tf.layers.conv2d(
            inputs=drop,
            filters=10,
            kernel_size=[20,EMBEDDING_DIMENSION//2],
            padding='SAME',# Add a ReLU for non linearity.
            activation=tf.nn.relu)
        # Max pooling across output of Convolution+Relu.
        pool2 = tf.layers.max_pooling2d(
            conv2,
            pool_size=[1, 4],
            strides=[1, 4],
            padding='VALID')
        # Max across each filter to get useful features for classification.
        #pool3 = tf.squeeze(pool2)
        flat = tf.reshape(drop, [-1, (MAX_SEQUENCE_LENGTH)*(EMBEDDING_DIMENSION//8)*10])
    with tf.variable_scope('fully_connected_2') as scope:
        # Apply regular WX + B and classification.
        fullyconnectedLayer = tf.layers.dense(flat, 15, activation=None)
        drop = tf.layers.dropout(fullyconnectedLayer, rate=0.5)
        output = tf.layers.dense(inputs=drop, units=1, activation=tf.nn.sigmoid, name=scope.name)


    return output

def lr(epoch):
    learning_rate = 1e-3
    if epoch > 80:
        learning_rate *= 0.5e-3
    elif epoch > 60:
        learning_rate *= 1e-3
    elif epoch > 40:
        learning_rate *= 1e-2
    elif epoch > 20:
        learning_rate *= 1e-1
    return learning_rate

train = pd.read_csv("labeledTrainData.tsv",header=0,delimiter="\t",quoting=3)
#
# unlabeled_train = pd.read_csv("unlabeledTrainData.tsv",
#                               header=0,
#                               delimiter="\t",
#                               quoting=3 )
#
test = pd.read_csv("testData.tsv",
                   header=0,
                   delimiter="\t",
                   quoting=3 )

X_train,X_test,VOCAB_SIZE,MAX_SEQUENCE_LENGTH = TextUtility.cleanReviewAndSequenize(train["review"],test["review"])
Y_train = np.array(train["sentiment"])
#Y_test = np.array(test["sentiment"])

with tf.name_scope('inputs'):
    data = tf.placeholder(tf.int32, name='inputs')

with tf.name_scope('labels'):
    labels = tf.placeholder(tf.int32, name='labels')

output = cnn_model(data, labels, VOCAB_SIZE=VOCAB_SIZE,EMBEDDING_DIMENSION=16,MAX_SEQUENCE_LENGTH=MAX_SEQUENCE_LENGTH)
# Calculate the cost
with tf.name_scope('cost'):
    cost = tf.losses.mean_squared_error(labels, output)
    tf.summary.scalar('cost', cost)

# Train the model
with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

# Determine the accuracy
with tf.name_scope("accuracy"):
    correct_pred = tf.equal(tf.cast(tf.round(output), tf.int32), labels)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
numberOfEpochs = 10
#print(X_train)
with tf.Session() as sess:
    print("starting the training")
    sess.run(tf.global_variables_initializer())
    for epoch in range(numberOfEpochs):
        print("working on epoch : ", epoch , "/" , numberOfEpochs)
        batch_size = X_train.shape[0]//32 # total reviews/no. of batches
        for currBatchNumber in range(32):
            batch_xs = X_train[ currBatchNumber * batch_size : (currBatchNumber + 1) * batch_size]
            batch_ys = Y_train[ currBatchNumber * batch_size : (currBatchNumber + 1) * batch_size]
            # print((currBatchNumber * batch_size) , " -- " ,((currBatchNumber + 1) * batch_size) )
            # print(batch_xs)
            # print(batch_ys)
            _, loss, acc = sess.run([optimizer,cost,accuracy], feed_dict= {data: batch_xs, labels: batch_ys})
            print( "Batch no: {} -- Accuracy: {} -- Loss: {}".format(currBatchNumber,acc,loss))


# # Process vocabulary
# vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(MAX_DOCUMENT_LENGTH)
# x_train = np.array(list(vocab_processor.fit_transform(x_train)))
# x_test = np.array(list(vocab_processor.transform(x_test)))
# n_words = len(vocab_processor.vocabulary_)
# print('Total words: %d' % n_words)
