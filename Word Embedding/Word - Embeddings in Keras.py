# the below program will shows that the embedding layer in keras do not take into consideration the semantic relationship
# between the words.
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Convolution1D, Flatten
from keras.layers.embeddings import Embedding
from keras import backend as K



# training set. Contains a row of size 5 per train example. The row is same as a sentence, with words replaced
# by its equivalent unique index. The below dataset contains 6 unique words numbered 0-5. Ideally the word vector for
# 4 and 5 indexed words should be same.
X_train = np.array([[0,1,4,2,3],[0,1,5,2,3]])

# output dummy for testing purpose
y_train = np.array([0,1])


# create the model
model = Sequential()

# there are 6 unique words and we want a 3 sized vector per word. Input_length if the max size of each input sentence.
temp = model.add(Embedding(6, 3, input_length=5))

# model
model.add(Convolution1D(64, 3, padding='same'))
model.add(Flatten())
model.add(Dense(180,activation='sigmoid'))
model.add(Dense(1,activation='sigmoid'))

# Compiling the model. For testing purpose we do not case about the optimizers and loss functions.
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the model on the training samples.
model.fit(X_train, y_train, epochs=3)


# the below proves that the embedding layer is not learning the semantic relationship between the words.
# it has just developed an unique weight matrix for each word.
get_3rd_layer_output = K.function([model.layers[0].input],[model.layers[0].output])
layer_output = get_3rd_layer_output([X_train])
print(layer_output)


#####OUTPUT#######
#Epoch 1/3

#2/2 [==============================] - 2s 829ms/step - loss: 0.7971 - acc: 0.5000
#Epoch 2/3

#2/2 [==============================] - 0s 1ms/step - loss: 0.7745 - acc: 0.5000
#Epoch 3/3

#2/2 [==============================] - 0s 1ms/step - loss: 0.7541 - acc: 0.5000
#[array([[[ 0.02520598,  0.04194991,  0.02291824],
#        [-0.01572827, -0.03382527,  0.05200452],
#        [-0.02286334, -0.02372537,  0.02930251],
#        [-0.03068878,  0.00251579,  0.04077445],
#        [ 0.02052142, -0.01374925,  0.01355451]],

#       [[ 0.02520598,  0.04194991,  0.02291824],
#        [-0.01572827, -0.03382527,  0.05200452],
#        [-0.00675715, -0.02608048, -0.05001082],
#        [-0.03068878,  0.00251579,  0.04077445],
#        [ 0.02052142, -0.01374925,  0.01355451]]], dtype=float32)]


