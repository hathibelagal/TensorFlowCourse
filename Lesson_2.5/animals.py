from tensorflow import keras
import numpy as np
import pandas as pd

training_data = pd.read_csv('zoo.csv')

output_data = np.array(
    training_data["class_type"].values
)

outputs = np.zeros((output_data.size, 8))
outputs[np.arange(output_data.size), output_data] = 1

inputs = np.array(
    training_data[training_data.columns.values[1:17]]   
)

my_network = keras.Sequential()

my_network.add(keras.layers.Dense(
    30,
    input_dim=16,
    activation='relu'
))

my_network.add(keras.layers.Dense(
    20,
    activation='relu'
))

my_network.add(keras.layers.Dense(
    8,
    activation='softmax'
))

my_network.compile(
    optimizer='sgd',
    loss='mse'
)

#my_network.fit(inputs, outputs, epochs=7500)
#my_network.save_weights('animals.h5')

my_network.load_weights('animals.h5')

labels = [None, "mammal", "bird", "reptile", "fish", "amphibian", "bug", "inverterbrate"]

#tiger
test_data = np.array([
    [1,0,0,1,0,0,1,1,1,1,0,0,4,1,0,1]
])

predictions = my_network.predict(test_data)

highestNeuron = np.argmax(predictions[0])
print "Animal is %s" % labels[highestNeuron]

keras.backend.clear_session()
