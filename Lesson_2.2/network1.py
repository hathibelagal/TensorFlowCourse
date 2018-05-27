from tensorflow import keras
import numpy as np

my_network = keras.Sequential()
my_network.add(keras.layers.Dense(
    12,
    input_dim=6,
    activation='relu'
))
my_network.add(keras.layers.Dense(
    8,
    activation='relu'
))
my_network.add(keras.layers.Dense(
    4,
    activation='softmax'
))
my_network.compile(
    optimizer='adam',
    loss='mse'
)

inputs=np.array([
    [1,0,0,1,0,0],
    [1,0,0,0,1,0],
    [1,0,0,0,0,1],
    [0,1,0,1,0,0],
    [0,1,0,0,1,0],
    [0,1,0,0,0,1],
    [0,0,1,1,0,0],
    [0,0,1,0,1,0],
    [0,0,1,0,0,1]
])

outputs=np.array([
    [0,1,0,0],
    [0,0,1,0],
    [0,1,0,0],
    [1,0,0,0],
    [0,0,1,0],
    [0,0,0,1],
    [1,0,0,0],
    [0,1,0,0],
    [0,0,0,1]
])

#my_network.fit(inputs, outputs, epochs=5000)
#my_network.save_weights('meals.h5')

my_network.load_weights('meals.h5')
print my_network.predict(np.array([
    [1,0,0,0,0,1]
]))

keras.backend.clear_session()
