import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import keras.backend as K
import os
from keras.utils import plot_model

top_words = 10000
max_review_length = 300
embedding_vector_length = 128

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

# Our data set is unbalanced, so, in addition to the regular accuracy, we will measure precision and recall.

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

# Model

model = Sequential()
model.add(Embedding(top_words, embedding_vector_length, input_length = max_review_length))
model.add(LSTM(100))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy', precision, recall])
print(model.summary())
plot_model(model, to_file='model.png')

# Training

model.fit(X_train, y_train, nb_epoch = 10, batch_size = 1024)

# Evaluation

evaluation_batch_size = y_test.shape[0]
scores = model.evaluate(X_test, y_test, verbose = 0, batch_size = evaluation_batch_size)
print("Performance on the test set:")
for i in range(len(scores)):
	print("%s: "%(model.metrics_names[i]), scores[i])

# Saving the model for future use

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")
print("The model was successfully saved")

