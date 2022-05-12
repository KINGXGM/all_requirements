from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import rmsprop_v2

model = Sequential()
model.add(Dense(units=4,input_shape=(4,)))
model.add(Dense(units=9))
model.add(Dense(units=20))
model.add(Dense(units=3,activation="softmax"))
model.compile(loss="categorical_crossentropy",optimizer=rmsprop_v2.RMSprop(learning_rate=0.01))
model.summary()
