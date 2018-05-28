
import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Convolution1D, Flatten, Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.callbacks import TensorBoard
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Flatten
from keras.layers.embeddings import Embedding


# Using keras to load the dataset with the top_words
top_words = 100000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)

print(X_train[3])
# Pad the sequence to the same length
max_review_length = 1600
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

# Shuffle data
shuffle_indices = np.random.permutation(np.arange(len(y_train)))
x = X_train[shuffle_indices]
y = y_train[shuffle_indices]
train_len = int(len(x) * 0.9)
X_train = x[:train_len]
y_train = y[:train_len]
X_val = x[train_len:]
y_val = y[train_len:]




model = Sequential()
e = Embedding(top_words, 200, input_length=max_review_length)
model.add(e)
model.add(Conv1D(filters=100, kernel_size=2, padding='valid', activation='relu', strides=1))
model.add(GlobalMaxPooling1D())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5, batch_size=32, verbose=2)

# Save the model to disk
model.save("trained_model.h5")
print("Model saved to disk.")

test_error_rate = model.evaluate(X_test, y_test, verbose=0)
print("The mean squared error (MSE) for the test data set is: {}".format(test_error_rate))


