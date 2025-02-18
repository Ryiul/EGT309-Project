import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


def ANN_train(X_train, X_test, y_train, y_test):
    model = Sequential()
    model.add(Dense(64,activation='relu'))
    model.add(Dense(32,activation='relu'))
    model.add(Dense(32,activation='relu'))
    model.add(Dense(1,activation='linear'))

    model.summary()

    model.compile(optimizer='adam',loss='mse',metrics=['mae'])

    history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test,y_test))

    model.save("model.h5")

    return model
