
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



# Using TensorFlow backend.
#
# Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz
# 17465344/17464789 [==============================] - 2s 0us/step
# [1, 4, 18609, 16085, 33, 2804, 4, 2040, 432, 111, 153, 103, 4, 1494, 13, 70, 131, 67, 11, 61, 15305, 744, 35, 3715, 761, 61, 5766, 452, 9214, 4, 985, 7, 64317, 59, 166, 4, 105, 216, 1239, 41, 1797, 9, 15, 7, 35, 744, 2413, 31, 8, 4, 687, 23, 4, 33929, 7339, 6, 3693, 42, 38, 39, 121, 59, 456, 10, 10, 7, 265, 12, 575, 111, 153, 159, 59, 16, 1447, 21, 25, 586, 482, 39, 4, 96, 59, 716, 12, 4, 172, 65, 9, 579, 11, 6004, 4, 1615, 5, 23005, 7, 5168, 17, 13, 7064, 12, 19, 6, 464, 31, 314, 11, 87564, 6, 719, 605, 11, 8, 202, 27, 310, 4, 3772, 3501, 8, 2722, 58, 10, 10, 537, 2116, 180, 40, 14, 413, 173, 7, 263, 112, 37, 152, 377, 4, 537, 263, 846, 579, 178, 54, 75, 71, 476, 36, 413, 263, 2504, 182, 5, 17, 75, 2306, 922, 36, 279, 131, 2895, 17, 2867, 42, 17, 35, 921, 18435, 192, 5, 1219, 3890, 19, 20523, 217, 4122, 1710, 537, 20341, 1236, 5, 736, 10, 10, 61, 403, 9, 47289, 40, 61, 4494, 5, 27, 4494, 159, 90, 263, 2311, 4319, 309, 8, 178, 5, 82, 4319, 4, 65, 15, 9225, 145, 143, 5122, 12, 7039, 537, 746, 537, 537, 15, 7979, 4, 18665, 594, 7, 5168, 94, 9096, 3987, 15242, 11, 28280, 4, 538, 7, 1795, 246, 56615, 9, 10161, 11, 635, 14, 9, 51, 408, 12, 94, 318, 1382, 12, 47, 6, 2683, 936, 5, 6307, 10197, 19, 49, 7, 4, 1885, 13699, 1118, 25, 80, 126, 842, 10, 10, 47289, 18223, 4726, 27, 4494, 11, 1550, 3633, 159, 27, 341, 29, 2733, 19, 4185, 173, 7, 90, 16376, 8, 30, 11, 4, 1784, 86, 1117, 8, 3261, 46, 11, 25837, 21, 29, 9, 2841, 23, 4, 1010, 26747, 793, 6, 13699, 1386, 1830, 10, 10, 246, 50, 9, 6, 2750, 1944, 746, 90, 29, 16376, 8, 124, 4, 882, 4, 882, 496, 27, 33029, 2213, 537, 121, 127, 1219, 130, 5, 29, 494, 8, 124, 4, 882, 496, 4, 341, 7, 27, 846, 10, 10, 29, 9, 1906, 8, 97, 6, 236, 11120, 1311, 8, 4, 23643, 7, 31, 7, 29851, 91, 22793, 3987, 70, 4, 882, 30, 579, 42, 9, 12, 32, 11, 537, 10, 10, 11, 14, 65, 44, 537, 75, 11876, 1775, 3353, 12716, 1846, 4, 11286, 7, 154, 5, 4, 518, 53, 13243, 11286, 7, 3211, 882, 11, 399, 38, 75, 257, 3807, 19, 18223, 17, 29, 456, 4, 65, 7, 27, 205, 113, 10, 10, 33058, 4, 22793, 10359, 9, 242, 4, 91, 1202, 11377, 5, 2070, 307, 22, 7, 5168, 126, 93, 40, 18223, 13, 188, 1076, 3222, 19, 4, 13465, 7, 2348, 537, 23, 53, 537, 21, 82, 40, 18223, 13, 33195, 14, 280, 13, 219, 4, 52788, 431, 758, 859, 4, 953, 1052, 12283, 7, 5991, 5, 94, 40, 25, 238, 60, 35410, 4, 15812, 804, 27767, 7, 4, 9941, 132, 8, 67, 6, 22, 15, 9, 283, 8, 5168, 14, 31, 9, 242, 955, 48, 25, 279, 22148, 23, 12, 1685, 195, 25, 238, 60, 796, 13713, 4, 671, 7, 2804, 5, 4, 559, 154, 888, 7, 726, 50, 26, 49, 7008, 15, 566, 30, 579, 21, 64, 2574]
# Train on 22500 samples, validate on 2500 samples
# Epoch 1/5
#  - 32s - loss: 0.3551 - acc: 0.8377 - val_loss: 0.2734 - val_acc: 0.8860
# Epoch 2/5
#  - 31s - loss: 0.1383 - acc: 0.9470 - val_loss: 0.3023 - val_acc: 0.8780
# Epoch 3/5
#  - 31s - loss: 0.0304 - acc: 0.9909 - val_loss: 0.4028 - val_acc: 0.8848
# Epoch 4/5
#  - 31s - loss: 0.0055 - acc: 0.9989 - val_loss: 0.4664 - val_acc: 0.8892
# Epoch 5/5
#  - 31s - loss: 8.3200e-04 - acc: 1.0000 - val_loss: 0.5193 - val_acc: 0.8848
# Model saved to disk.
# The mean squared error (MSE) for the test data set is: [0.5021236272479593, 0.88172]


