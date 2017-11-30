import numpy as np
from keras.models import Sequential, model_from_json
from keras.layers import LSTM, Dense, Dropout
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import keras.backend as K
import os

max_review_length = 300

X = np.load('xs.npy', encoding = 'latin1')
y = np.load('ys.npy', encoding = 'latin1')

np.random.seed(7)
m = y.shape[0]
data = np.hstack([np.reshape(y, (m, 1)), np.reshape(X, (m, 1))])
np.random.shuffle(data)
np.random.seed()

X = data[:, 1]
y = data[:, 0]

X_train = X[:int(0.8*m)]
y_train = y[:int(0.8*m)]
X_test = X[int(0.8*m):]
y_test = y[int(0.8*m):]

X_train = sequence.pad_sequences(X_train, maxlen = max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen = max_review_length)

def precision(y_true, y_pred):
	true_positives = K.sum(K.round(K.clip(y_true*y_pred, 0, 1)))
	predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
	precision = true_positives/(predicted_positives+K.epsilon())
	return precision

def recall(y_true, y_pred):
	true_positives = K.sum(K.round(K.clip(y_true*y_pred, 0, 1)))
	possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
	recall = true_positives/(possible_positives+K.epsilon())
	return recall

json_file = open('model.json', 'r')
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("model.h5")
model.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy', precision, recall])
print(model.summary())

evaluation_batch_size = y_test.shape[0]
scores = model.evaluate(X_test, y_test, verbose = 0, batch_size = evaluation_batch_size)
print("Performance on the test set:")
for i in range(len(scores)):
	print("%s: "%(model.metrics_names[i]), scores[i])


