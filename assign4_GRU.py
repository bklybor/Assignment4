from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten, Dropout, GRU
from keras.datasets import imdb
import matplotlib.pyplot as plt
from TimeHistory import TimeHistory
import numpy as np

batch_size = 128
epochs = 20
max_features = 500
maxlen=50

maxDiv = 10
dropouts = []
accuracies = []
scores = []
trainingTimes = []
time_callback = TimeHistory()
for i in range(0,maxDiv):
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

    model = Sequential()
    dropout = i/maxDiv
    model.add(Embedding(max_features, 128))
    model.add(GRU(128, dropout = dropout))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

    print('Train...')
    model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_test, y_test))
    times = time_callback.times
    trainingTimes.append(np.average(times))
    score, acc = model.evaluate(x_test, y_test,
                                batch_size=batch_size)
    print('Test score:', score)
    print('Test accuracy:', acc)
    dropouts.append(dropout)
    accuracies.append(acc)
    scores.append(score)


plt.plot(dropouts, accuracies, label='accuracies')
plt.xlabel('Dropout Rate')
plt.ylabel('Accuracy')
plt.legend()
plt.show()