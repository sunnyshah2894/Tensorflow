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