#!/usr/bin/env python
import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Convolution1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load the dataset but only keep the top n words, zero the rest
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=top_words)

# Pad the small reviews
max_words = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)

def create_model(pool_length, nb_epoch, hidden_layer_size=1, init_mode='uniform'):
    
    #print("Testing new model")
    
    # create the model
    model = Sequential()
    model.add(Embedding(top_words, 32, input_length=max_words))
    model.add(Convolution1D(nb_filter=32, filter_length=3, border_mode='same', activation='relu'))
    model.add(MaxPooling1D(pool_length=pool_length))
    model.add(Flatten())
    model.add(Dense(hidden_layer_size, init=init_mode, activation='relu'))
    model.add(Dense(1, init=init_mode, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    #print(model.summary())
    
    return model

# create model
model = KerasClassifier(build_fn=create_model, verbose=2)

# define the grid search parameters
batch_size = [10, 20, 40, 60, 80, 100, 128]
epochs = [2, 10]
pool_length = [2, 5, 10, 20, 32, 500]
hidden_layer_size = [32,250, 500]
init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']

param_grid = dict(batch_size=batch_size, nb_epoch=epochs, pool_length=pool_length, hidden_layer_size=hidden_layer_size, init_mode=init_mode)

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, verbose=10)

grid_result = grid.fit(X_train, y_train)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
